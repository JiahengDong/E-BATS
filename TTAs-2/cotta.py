from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC
from time import time
import logging
from jiwer import wer
from data import load_dataset
import torchaudio
from audiomentations import Compose, AddGaussianNoise, PitchShift
import numpy as np
import os
import random

def set_seed(seed=42):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def generate_pseudo_labels(input_values, model, processor):
    """
    Generate pseudo-labels (transcriptions) for the given audio input using the model.
    """
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription  # Return the pseudo-transcription labels

class CoTTA(nn.Module):
    def __init__(self, model, processor, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)  

        # Stochastic restoration parameters
        self.alpha_teacher = 0.999
        self.stochastic_restore_prob = 0.001  # Probability of restoring weights
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
        ])

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)                         


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        inputs = self.processor(x, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        
        outputs = self.model(input_values).logits #student model prediction
        student_trans = self.processor.batch_decode(torch.argmax(outputs, dim=-1))
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(input_values).logits, dim=-1).max(-1)[0]
        standard_ema = self.model_ema(input_values).logits
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        to_aug = anchor_prob.mean(dim=[0,1]) < 0.2
        if to_aug: 
            for i in range(N):
                wav = np.array(x)
                augmented_inputs = self.augment(wav, SAMPLE_RATE)
                for augmented_input in augmented_inputs:
                    augmented_input = self.processor(augmented_input, return_tensors="pt", padding="longest")
                    augmented_input_values = augmented_input.input_values.cuda()
                    outputs_  = self.model_ema(augmented_input_values.cuda()).logits.detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        loss = (softmax_entropy(outputs, outputs_ema.detach()))
        student_predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(student_predicted_ids != 0, 1, 0).bool()
        loss = loss * non_blank.float()  # Apply mask, keeping only non-blank entropies
        loss = loss.sum(dim=1) / non_blank.sum(dim=1).clamp(min=1)  # Mean entropy for each sequence
        loss = loss.mean(0).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        if loss.item() > 0:
            self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.alpha_teacher)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        torch.manual_seed(42)
                        mask = (torch.rand(p.shape)<0.001).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

        
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(-1) * x.log_softmax(-1)).sum(-1)-0.5*(x.softmax(-1) * x_ema.log_softmax(-1)).sum(-1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    for m in model.modules():
        m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def setup_cotta(model, processor, steps, episodic):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = CoTTA(model, processor, optimizer,
                           steps=steps,
                           episodic=episodic)
 
    return cotta_model


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    elif opt_name == "SGD":
        optimizer = opt(params,
                lr=0.01,
                momentum=0.9,
                dampening=0,
                weight_decay=weight_decay,
                nesterov=True)
    else: 
        optimizer = opt(params,
            lr=lr,
            betas=(beta, 0.999),
            weight_decay=weight_decay,
            foreach=False
            )
    
    return optimizer

import argparse

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--extra_noise', type=float, default=0.0)

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    episodic = args.episodic
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    lr = args.lr
    batch_size = args.batch_size
    opt = args.opt
    extra_noise = args.extra_noise

    if args.split == 'overall':
        split = ['et05_bus_simu', 'et05_bus_real', 'et05_caf_real', 'et05_caf_simu', 'et05_ped_real', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu']
    elif args.split == 'simu':
        split = ['et05_bus_simu', 'et05_caf_simu', 'et05_ped_simu', 'et05_str_simu']
    elif args.split == 'real':
        split = ['et05_bus_real', 'et05_caf_real', 'et05_ped_real', 'et05_str_real']
    elif args.split == "bus-real":
        split = ['et05_bus_real']
    elif args.split == "bus-simu":
        split = ['et05_bus_simu']
    elif args.split == "ped-real":
        split = ['et05_ped_real']
    elif args.split == "caf-real":
        split = ['et05_caf_real']
    elif args.split == "str-real":
        split = ['et05_str_real']
    elif args.split == "caf-simu":
        split = ['et05_caf_simu']
    elif args.split == "ped-simu":
        split = ['et05_ped_simu']
    elif args.split == 'str-simu':
        split = ['et05_str_simu']

    set_seed(42) 
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)

    transcriptions_1 = []
    transcriptions_3 = []
    transcriptions_5 = []
    transcriptions_10 = []

    gt_texts = []
    ori_transcriptions = []
    durations = []
    werrs = []

    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    if asr == "facebook/wav2vec2-base-960h":
        # load model and tokenizer
        base_model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
    elif asr == "facebook/hubert-large-ls960-ft":
        base_model = HubertForCTC.from_pretrained(asr).eval().cuda()
    model = setup_cotta(base_model, processor, steps, episodic)

    count = 0

    import time
    start = time.time()
    memory_usage = []
    for batch in dataset:
        count += 1
        lens, wavs, texts, files = batch

        torch.cuda.reset_peak_memory_stats()
        for i in range(steps):
            outputs = model.forward_and_adapt(wavs)
        
        predicted_ids = torch.argmax(outputs, dim=-1)        
        transcription = processor.batch_decode(predicted_ids)           
        ada_wer = wer(list(texts), list(transcription))
        #print("adapt WER:  ", ada_wer)
        #print(texts, transcription)
        transcriptions_1 += transcription
            
        
        memory_usage.append(torch.cuda.max_memory_allocated()/(1024*1024))
        #print("Max memory usage:", max(memory_usage))
        torch.cuda.empty_cache()
        gt_texts += texts


    print("asr:", asr)
    print(f'non-adapted count = {count}')
    print(f'dataset num = {len(dataset)}')
    print("WER:", wer(gt_texts, transcriptions_1))
    print("Overall Max memory usage:", max(memory_usage))