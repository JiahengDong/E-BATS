import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC
from torch import nn
from jiwer import wer
import cma
import math
import numpy as np
from data import load_dataset
import time
import random
import argparse
import json

def set_seed(seed=42):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class Wav2Vec2WithPrompts(nn.Module):
    def __init__(self, model_name, num_prompts, processor, popsize, random_seed, 
    dataset_name, alpha, beta, temp, tokenwise, reset_frequency, ema_decay, use_tema, confidence_max, covariance_ratio):
        super(Wav2Vec2WithPrompts, self).__init__()
        if model_name == 'facebook/wav2vec2-base-960h':
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).eval().cuda()
            self.base_model = self.model.wav2vec2
        elif model_name == 'facebook/hubert-large-ls960-ft':
            self.model = HubertForCTC.from_pretrained(model_name).eval().cuda()
            self.base_model = self.model.hubert

        self.model.requires_grad_(False)
        self.base_model.requires_grad_(False)
        self.prompts = nn.ParameterList()
        self.processor = processor
        self.hist_stat = None
        self.tokenwise_all_hidden = {}

        self.count = 0
        self.popsize = popsize
        self.steps = 0
        self.dataset_name = dataset_name
        self.tokenwise = tokenwise
        self.sigma = 0.01
        self.random_seed = random_seed
        self.confidence_max = confidence_max
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

        for i, layer in enumerate(self.base_model.feature_extractor.conv_layers):
            if i == 6:
                output_channel = layer.conv.out_channels
                beta = nn.Parameter(torch.zeros(1, output_channel, num_prompts, device='cuda'))
                fan_in = 512
                fan_out = output_channel
                val = math.sqrt(6. / float(fan_in + fan_out))
                nn.init.uniform_(beta.data, -val, val)
                self.prompts.append(beta)
    
        self.num_prompts = num_prompts
        self.best_loss = np.inf
        self.es = self.init_cma()
        self.initial_C = self.es.C.copy()  # Store the initial covariance matrix
        
        # Initialize EMA versions of CMA-ES components
        self.ema_C = self.es.C.copy()
        self.ema_pc = np.zeros(self.es.N)
        self.ema_sigma = self.es.sigma
        self.ema_ps = np.zeros(self.es.N)
        self.ema_mean = np.zeros(self.es.N)
        self.use_tema = use_tema
        self.ema_decay = ema_decay
        self.covariance_ratio = covariance_ratio

        self.reset_counter = 0  # Counter to track number of utterances processed
        self.reset_frequency = reset_frequency  # Reset after every utterance by default
        self.train_info = self.compute_in_domain_statistics_feature_extractor()

                                
    def init_cma(self):
        """CMA-ES initialization"""
        dim = sum(p.numel() for p in self.prompts)
        popsize = self.popsize
        sigma = self.sigma
        cma_opts = {
            'seed': self.random_seed,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0.0], sigma, inopts=cma_opts)
        self.popsize = es.popsize
        return es

    def compute_in_domain_statistics_feature_extractor(self):
        dataset = load_dataset(['test-clean'], 'librispeech', '/home/jiahengd/tta-suta/LibriSpeech', 1, 0.)
        all_feat_hidden = []
        all_trans_hidden = []
        tokenwise_hidden_map = {}
        num_batches = 0

        for i, batch in enumerate(dataset):
            num_batches += 1
            # Assuming the batch contains raw waveforms
            lens, wav, texts, files = batch
            inputs = self.processor(wav, return_tensors="pt", padding="longest")
            input_values = inputs.input_values.cuda()

            with torch.no_grad():
                # Extract feature extractor hidden states
                feat_hidden = self.base_model.feature_extractor(input_values)
                feat_hidden = feat_hidden.transpose(1,2)
                processed_feat_hidden = feat_hidden.mean(dim=1) 
                
                # Extract transformer encoder hidden states
                outputs = self.base_model(input_values, output_hidden_states = True)
                trans_hidden = outputs.hidden_states
                processed_trans_hidden = torch.cat([hidden_state.mean(dim=1) for hidden_state in trans_hidden], dim=1)
                
                all_feat_hidden.append(processed_feat_hidden)
                all_trans_hidden.append(processed_trans_hidden)
                
                # Extract tokenwise hidden states
                outputs_logits = self.model(input_values).logits
                predicted_ids = torch.argmax(outputs_logits, dim=-1) 
                for j in range(predicted_ids.size(0)):
                    tokens = predicted_ids[j]
                    for k, token in enumerate(tokens):
                        token_id = int(token.item())
                        if token_id == self.processor.tokenizer.pad_token_id:
                            continue
                        if token_id not in tokenwise_hidden_map:
                            tokenwise_hidden_map[token_id] = []
                        tokenwise_hidden_map[token_id].append(torch.cat([h[j][k] for h in trans_hidden], dim = 0))

        # calculate tokenwise source statistics
        for token_id, token_features in tokenwise_hidden_map.items():
            token_all_features = torch.stack(token_features)
            token_std, token_mean = torch.std_mean(token_all_features, dim = 0, unbiased=False)
            self.tokenwise_all_hidden[token_id] = {'token_std': token_std, 'token_mean': token_mean}

        # Concatenate all hidden states across batches
        all_feat_hidden = torch.stack(all_feat_hidden).squeeze(1)
        all_trans_hidden = torch.stack(all_trans_hidden).squeeze(1) 
        # Calculate utterance level source statistics
        feat_batch_std, feat_batch_mean = torch.std_mean(all_feat_hidden, dim=0, unbiased=False)
        trans_batch_std, trans_batch_mean = torch.std_mean(all_trans_hidden, dim=0, unbiased=False)

        # Clear temporary data to free memory
        del all_feat_hidden, all_trans_hidden, tokenwise_hidden_map
        torch.cuda.empty_cache()
        
        print(f"Computed source statistics from {num_batches} samples")
        return [feat_batch_mean, feat_batch_std, trans_batch_mean, trans_batch_std]

    def forward(self, x):
        # Ensure input_values is on the same device as the model
        self.base_model.config.output_hidden_states = True
        x = x[:, None]
        for i, layer in enumerate(self.base_model.feature_extractor.conv_layers):
            x = layer.conv(x)
            if i == 6: 
                beta = self.prompts[0]
                x = x + beta
            if hasattr(layer, "layer_norm"):
                # Check if it's actually LayerNorm (Hubert) or GroupNorm (Wav2Vec2)
                if isinstance(layer.layer_norm, torch.nn.LayerNorm):
                    # Hubert LayerNorm: expects (batch, length, features)
                    x = x.transpose(-2,-1)
                    x = layer.layer_norm(x)
                    x = x.transpose(-2,-1)
                else:
                    # Wav2Vec2 GroupNorm: expects (batch, features, length)
                    x = layer.layer_norm(x)
            if hasattr(layer, "activation"):
                x = layer.activation(x)

        x = x.transpose(1,2)
        feat_hidden = x
        # Handle different return types for wav2vec2 vs hubert
        if hasattr(self.model, 'wav2vec2'):
            # Wav2Vec2 returns (hidden, features)
            pro_feat_hidden, _ = self.base_model.feature_projection(x)
        else:
            # HuBERT returns only hidden
            pro_feat_hidden = self.base_model.feature_projection(x)
        
        # Transformer encoder - uses the same number of layers as the original Wav2Vec2 model
        with torch.no_grad():
            transformer_output = self.base_model.encoder(pro_feat_hidden, output_hidden_states=True)
        logits = self.model.lm_head(transformer_output.last_hidden_state)
        return logits, feat_hidden, transformer_output.hidden_states

    def update_ema_components(self):
        """Update EMA of CMA-ES components based on current state"""
        self.ema_C = self.ema_decay * self.ema_C + (1 - self.ema_decay) * self.es.C
        self.ema_pc = self.ema_decay * self.ema_pc + (1 - self.ema_decay) * self.es.pc
        self.ema_sigma = self.ema_decay * self.ema_sigma + (1 - self.ema_decay) * self.es.sigma
        if hasattr(self.es, 'ps'):
            self.ema_ps = self.ema_decay * self.ema_ps + (1 - self.ema_decay) * self.es.ps      
        self.ema_mean = self.ema_decay * self.ema_mean + (1 - self.ema_decay) * self.es.mean

    def reset_cma_components_original(self):
        """Reset CMA-ES to its initial values"""
        self.es = self.init_cma()
        self.reset_counter = 0

    def reset_cma_components_tema(self):
        """Reset CMA-ES using TEMA-style strategy with EMA values
        
        This method always:
        - Uses EMA mean
        - Blends EMA covariance with identity matrix (50/50)
        - Resets evolution paths (pc and ps)
        - Keeps the step size (sigma) from EMA
        """
        print("Performing TEMA-style reset of CMA-ES")
        
        # Apply the TEMA-style reset exactly as specified
        self.es.mean = self.ema_mean.copy()
        self.es.sigma = self.ema_sigma
        self.es.C = self.covariance_ratio * self.ema_C + (1 - self.covariance_ratio) * np.eye(len(self.ema_mean))  # partial covariance reset
        self.es.pc = np.zeros_like(self.ema_mean)
        self.es.ps = np.zeros_like(self.ema_mean)
        # Recompute eigendecomposition
        self.es.dC = np.diag(np.diag(self.es.C))
        self.es.D, self.es.B = np.linalg.eigh(self.es.C)
        self.es.D = np.sqrt(self.es.D)
        self.reset_counter = 0
    
    def reset_cma_components(self):
        """Wrapper function to call the appropriate reset method based on reset type"""
        if hasattr(self, 'use_tema') and self.use_tema:
            self.reset_cma_components_tema()
        else:
            self.reset_cma_components_original()

def softmax_entropy(x, dim=-1):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def forward_and_adapt(x, model):
    model.best_loss, best_outputs = np.inf, None
    """Sampling from CMA-ES and evaluate the new solutions."""
    prompts, losses = model.es.ask(), []
    for j, prompt in enumerate(prompts):
        start = 0
        for p in model.prompts:
            length = p.numel()
            p.data.copy_(torch.tensor(prompt[start:start + length], dtype=torch.float).reshape_as(p).cuda())
            start += length
        model.prompts.requires_grad_(False)
        outputs, loss = forward_and_get_loss(x, model)

        if model.best_loss > loss.item():
            model.best_loss = loss.item()
            best_outputs = outputs
        losses.append(loss.item())
        del outputs

    prompts = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in prompts]
   
    model.es.tell(prompts, losses)
    return best_outputs
    

def forward_and_get_loss(x, model):
    outputs, feat_hidden, trans_hidden = model.forward(x)
    processed_feat_hidden = feat_hidden.mean(dim=1)
    processed_trans_hidden = torch.cat([hidden_state.mean(dim=1) for hidden_state in trans_hidden], dim=1)

    feat_source_mean = model.train_info[0].cuda()
    feat_source_std = model.train_info[1].cuda()
    trans_source_mean = model.train_info[2].cuda()
    trans_source_std = model.train_info[3].cuda()

    # Calculate feature extractor output target statistics
    criterion_mse = nn.MSELoss(reduction='none').cuda()
    feat_target_std, feat_target_mean = torch.std_mean(processed_feat_hidden, dim=0, unbiased=False)
    feat_std_mse, feat_mean_mse = criterion_mse(feat_target_std, feat_source_std), criterion_mse(feat_target_mean, feat_source_mean)
    feat_utterance_loss = feat_mean_mse.sum()

    # Calculate transformer encoder target statistics
    trans_target_std, trans_target_mean = torch.std_mean(processed_trans_hidden, dim=0, unbiased=False)
    trans_std_mse, trans_mean_mse = criterion_mse(trans_target_std, trans_source_std), criterion_mse(trans_target_mean, trans_source_mean)
    # Average trans_mean_mse by hidden_size because transformer differences are much larger than feature extractor differences.
    # This prevents trans_utterance_loss from dominating and overwhelming feat_utterance_loss in the final loss calculation.
    trans_utterance_loss = trans_mean_mse.sum() / model.base_model.config.hidden_size

    predicted_ids = torch.argmax(outputs, dim=-1)
    non_blank = torch.where(predicted_ids != 0, 1, 0).bool() 

    if non_blank.sum() == 0:
        entropy_loss = torch.tensor(10, device=outputs.device)
    else:
        entropy_loss = softmax_entropy(outputs / model.temp)[non_blank].mean(0).mean()
    
    loss = model.alpha*entropy_loss + model.beta*(feat_utterance_loss + trans_utterance_loss)

    if model.tokenwise:
        token_target_features = {}
        token_std_diff = 0
        token_mean_diff = 0
        count = 0
        for i, tokens in enumerate(predicted_ids):
            tokens_id = predicted_ids[i]
            for j, token in enumerate(tokens_id):
                token_id = int(token.item())
                if token_id != 0:
                    if token_id == model.processor.tokenizer.pad_token_id:
                        continue
                    if token_id not in token_target_features:
                        token_target_features[token_id] = []
                    # Detach to prevent memory leak from computation graph
                    token_target_features[token_id].append(torch.cat([h[i][j].detach() for h in trans_hidden], dim=0))

        for token, token_features in token_target_features.items():
            count+=1
            all_token_target_features = torch.stack(token_features)
            token_target_std, token_target_mean = torch.std_mean(all_token_target_features, dim = 0, unbiased=False)
            token_std_mse, token_mean_mse = criterion_mse(model.tokenwise_all_hidden[token]['token_std'], token_target_std), criterion_mse(model.tokenwise_all_hidden[token]['token_mean'], token_target_mean)
            token_mean_diff += token_mean_mse.mean()
            token_std_diff += token_std_mse.mean()

        if count == 0:
            tokenwise_loss = torch.tensor(0)
        else:
            tokenwise_loss = (token_mean_diff + token_std_diff) / count
        
        tokenwise_scale = confidence_scaling(loss, 0.0, model.confidence_max)
        loss += tokenwise_scale * tokenwise_loss
    
    return outputs, loss

def confidence_scaling(entropy_val, ent_min=0.0, ent_max=6.0):
    """
    Suppose entropy_val ~ [0, ent_max].
    We can map high entropy -> small weight,
               low entropy -> bigger weight.
    """
    norm = (entropy_val - ent_min) / (ent_max - ent_min + 1e-8)
    norm = max(0.0, min(2.0, norm))
    return 2.0 - norm

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--tokenwise', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/jiahengd/tta-suta/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pop_size', type=int, default=50)
    parser.add_argument('--early_stop_threshold', type=float, default=0.001)
    parser.add_argument('--patient', type=int, default=3)
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--temp', type=float, default=2.0)
    # args for LS extra noise 
    parser.add_argument('--reset_frequency', type=int, default=1, 
                        help='Reset CMA-ES components after this many utterances (default: 1 = reset every utterance)')
    parser.add_argument('--ema_decay', type=float, default=0.9,
                        help='EMA decay rate for CMA-ES component tracking (default: 0.9)')
    parser.add_argument('--use_tema', action='store_true',
                        help='Use TEMA-style reset of CMA-ES components instead of EMA values')
    parser.add_argument('--covariance_ratio', type=float, default=0.5,
                        help='Covariance ratio for TEMA-style reset (default: 0.5)')
    parser.add_argument('--random_seed', type=int, default=2024,
                        help='Random seed for CMA-ES (default: 2024)')
    parser.add_argument('--confidence_max', type=float, default=5.0,
                        help='Maximum confidence value for confidence scaling (default: 5.0)')

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    batch_size = args.batch_size
    popsize = args.pop_size
    early_stop_threshold = args.early_stop_threshold
    patient = args.patient
    extra_noise = args.extra_noise
    random_seed = args.random_seed
    confidence_max = args.confidence_max
    covariance_ratio = args.covariance_ratio
    tokenwise = args.tokenwise
    reset_frequency = args.reset_frequency
    ema_decay = args.ema_decay
    use_tema = args.use_tema
    alpha = args.alpha
    beta = args.beta
    temp = args.temp

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

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    adpat_model = Wav2Vec2WithPrompts(asr, 1, processor, popsize, random_seed, dataset_name, alpha, beta, temp, 
                                    tokenwise, reset_frequency, ema_decay, use_tema, confidence_max, covariance_ratio)
    
    
    utterance_counter = 0  # Track number of utterances processed
    gt_texts = []
    transcriptions = []
    memory_usage = []
    utterance_duration = []
    wers = []

    start = time.time()
    for batch in dataset:
        
        adpat_model.steps += 1
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        
        # Define how many iterations we wait without improvement before stopping
        patience = patient
        not_improved_count = 0

        torch.cuda.reset_peak_memory_stats()
        for i in range(steps): 
            # Store the old "best loss" before we run this iteration
            old_loss = adpat_model.best_loss

            outputs = forward_and_adapt(input_values, adpat_model)
            
            #print(f'memory usage for single step, step: {i+1}, memory: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

            # Compare the updated best_loss with the old one
            loss_diff = old_loss - adpat_model.best_loss
            if old_loss - adpat_model.best_loss  > early_stop_threshold:
            # We got an improvement
                not_improved_count = 0
            else:
                # No improvement this iteration
                not_improved_count += 1
            
            if not_improved_count >= patience:
                print(f"Early stopping triggered at step {i+1}/{steps}, start breaking")
                break
        
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcriptions.extend(transcription)
        
        #collect wer and memory usage
        memory_usage.append(torch.cuda.max_memory_allocated()/(1024*1024))
        utterance_duration.append(input_values.shape[1]/SAMPLE_RATE)
        wers.append(wer(list(texts), list(transcription)))
        
        del input_values
        torch.cuda.empty_cache()
        gt_texts.extend(texts)

       # Update EMA stats
        adpat_model.update_ema_components()
            
        # Check if we need to reset components (TEMA or reset to original)
        utterance_counter += 1
        if utterance_counter >= reset_frequency:
            adpat_model.reset_cma_components()
            utterance_counter = 0  # Reset the counter
    
    print("asr:", asr)
    print(f'dataset num = {len(dataset)}')
    final_wer_val = wer(gt_texts, transcriptions)
    print("Final WER: ", final_wer_val)
    print(f"max memory usage: {np.max(memory_usage)}MB")
    print(f"avg memory usage: {np.mean(memory_usage)}MB")
    print(f"min memory usage: {np.min(memory_usage)}MB")
    print(f"max utterance duration: {np.max(utterance_duration)}s")
    print(f"avg utterance duration: {np.mean(utterance_duration)}s")
    print(f"min utterance duration: {np.min(utterance_duration)}s")

    results = {
        'asr': asr,
        'dataset num': len(dataset),
        'memory_usage': memory_usage,
        'utterance_duration': utterance_duration,
        'wers': wers,
        'final_wer': final_wer_val,
        'avg_memory_usage': np.mean(memory_usage),
        'max_memory_usage': np.max(memory_usage),
        'min_memory_usage': np.min(memory_usage),
        'avg_utterance_duration': np.mean(utterance_duration),
        'max_utterance_duration': np.max(utterance_duration),
        'min_utterance_duration': np.min(utterance_duration),
    }

    # Save results to file
    if asr == 'facebook/wav2vec2-base-960h':
        asr = 'wav2vec2-base-960h'
    elif asr == 'facebook/hubert-large-ls960-ft':
        asr = 'hubert-large-ls960-ft'

    output_file = f'/home/jiahengd/tta-suta/Accuracy-efficency-results/{asr}_{dataset_name}_{split}_{tokenwise}_{ema_decay}'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")