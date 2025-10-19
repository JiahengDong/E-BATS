import os 
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC
from torch import nn
from jiwer import wer
import math
import numpy as np
from sam import SAM
import torch
import random

# Global parameters for tracking statistics
class Stats:
    def __init__(self):
        self.filter_1 = 0
        self.filter_2 = 0

stats = Stats()

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
    
def setup_optimizer(params, opt_name='SAM', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(params, base_optimizer, lr, weight_decay=weight_decay, foreach=False)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    trainable = []
    if bias_only:
        trainable = ['bias']
    else: 
        trainable = ['weight', 'bias']

    
    for nm, m in model.named_modules():
        print(nm)
        if train_LN: 
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
            

    return params, names


import torch.nn.functional as F
from copy import deepcopy
def copy_model_and_optimizer(model, optimizer, scheduler):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None

def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    return model
    
def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data
        
def forward_and_adapt(x, model, optimizer, not_blank=True, scheduler=None, skip_short_thd=None, margin=0.4*math.log(32), reset_constant=0.2, ema = None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward, adjust to the ASR, especially CTC loss, filter out blank tokens
    outputs = model(x).logits
    predicted_ids = torch.argmax(outputs, dim=-1)
    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    entropys = entropys * non_blank.float()  # Apply mask, keeping only non-blank entropies
    entropys = entropys.sum(dim=1) / non_blank.sum(dim=1).clamp(min=1)  # Mean entropy for each sequence
    filter_ids_1 = torch.where(entropys < margin)
    if len(filter_ids_1[0]) == 0:
        stats.filter_1 += 1
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()
    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation

    # Apply the non-blank mask again
    entropys2 = softmax_entropy(model(x).logits) 
    entropys2 = entropys2 * non_blank.float()  # Apply mask
    entropys2 = entropys2.sum(dim=1) / non_blank.sum(dim=1).clamp(min=1)  # Mean entropy for each sequence

    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    if len(filter_ids_2[0]) == 0:
        stats.filter_2 += 1
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.3:
            print("ema < 0.3, now reset the model")
            reset_flag = True

    with torch.no_grad():
        outputs = model(x).logits

    return outputs, ema, reset_flag

import argparse



if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='SAM')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=1.)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    episodic = args.episodic
    opt = args.opt
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    lr = args.lr
    em_coef = args.em_coef
    reweight = args.reweight
    batch_size = args.batch_size
    temp =  args.temp
    non_blank = args.non_blank
    log_dir = args.log_dir
    extra_noise = args.extra_noise
    scheduler = args.scheduler
    div_coef = args.div_coef
    bias_only = args.bias_only
    train_feature = args.train_feature
    train_all = args.train_all
    skip_short_thd = None
    train_LN = True

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

    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
    transcriptions_1 = []
    gt_texts = []
    ori_transcriptions = []
    durations = []
    werrs = []

    print('------------------------------------')
    print(f'eposidic? {episodic}')
    print(f'lr = {lr}')
    print(f'optim = {opt}')
    print(f'step = {steps}')
    print(f'em_coef = {em_coef}')
    print(f'reweight = {reweight}')
    print(f'batch size = {batch_size}')
    print(f'temperature = {temp}')
    print(f'non_blank = {str(non_blank)}')
    print(f'extra_noise = {extra_noise}')
    print(f'scheduler = {str(scheduler)}')
    print(f'div_coef = {str(div_coef)}')
    print(f'bias_only = {bias_only}')
    print(f'train_feature = {train_feature}')
    print(f'train_all = {train_all}')
    print(f'train_LN = {train_LN}')

    if asr == "facebook/wav2vec2-base-960h":
        model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
    elif asr == "facebook/hubert-large-ls960-ft":
        model = HubertForCTC.from_pretrained(asr).eval().cuda()

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)       
    
    # set up for tent
    model = configure_model(model)
    params, param_names = collect_params(model, bias_only, train_feature, train_all, train_LN)
    optimizer, scheduler = setup_optimizer(params, opt, lr, scheduler=scheduler)

    model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    import time
    start = time.time()
    ema = None
    memory_usage = []
    for batch in dataset:
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        duration = input_values.shape[1] / SAMPLE_RATE
        durations.append(duration)
        
        #SAR
        torch.cuda.reset_peak_memory_stats()
        for i in range(steps): 
            outputs, ema, reset_flag = forward_and_adapt(input_values, model, optimizer, non_blank, scheduler, ema = ema)
            if reset_flag == True:
                ema = None
                model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state)
            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            ada_wer = wer(list(texts), list(transcription))
            transcriptions_1 += transcription
        
        memory_usage.append(torch.cuda.max_memory_allocated()/(1024*1024))
        
        del input_values
        torch.cuda.empty_cache()
        gt_texts += texts


    print("asr:", asr)
    print(f'dataset num = {len(dataset)}')
    print("WER:", wer(gt_texts, transcriptions_1))
    print("Max memory usage:", max(memory_usage))
    
    print('------------------------------------')


    