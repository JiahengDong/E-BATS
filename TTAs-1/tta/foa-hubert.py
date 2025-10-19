import os 
import torch
import pandas as pd
from torch import nn
import cma
import math
import numpy as np
from data import load_dataset
from systems.GradientBasedSystem import GradientBasedSystem
from systems.loss import softmax_entropy
from .base import IStrategy

class Wav2Vec2WithPrompts(nn.Module):
    def __init__(self, model, num_prompts, processor):
        super(Wav2Vec2WithPrompts, self).__init__()
        self.model = model.eval().cuda()

        # Detect model type and get the base model
        if hasattr(self.model, 'hubert'):
            
            self.base_model = self.model.hubert
            self.prompt_dim = self.model.hubert.config.hidden_size
        else:
            
            self.base_model = self.model.wav2vec2
            self.prompt_dim = self.model.wav2vec2.config.hidden_size
            
        self.num_prompts = num_prompts
        self.layer_norm = nn.LayerNorm(self.prompt_dim).cuda()
        self.project_dim = nn.Linear(self.prompt_dim, self.prompt_dim).cuda()
        self.best_loss = np.inf
        self.hist_stat = None
        self.processor = processor
        self.encoder_source_hidden = []
        self.encoder_target_hidden = []
        self.train_info = self.compute_in_domain_statistics()
        self.fitness_lambda = 0.2

        # Xavier uniform initialization 
        fan_in = self.prompt_dim
        fan_out = self.prompt_dim
        val = math.sqrt(6. / float(fan_in + fan_out))
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)
        self.best_prompts = self.prompts
        self.es = self.init_cma()

    def init_promts(self):
        # Assuming fan_in could be the size of the input features (e.g., 80 for log-mel spectrograms)
        # fan_out is the prompt_dim, related to the transformer hidden size
        fan_in = self.base_model.config.conv_dim[-1] 
        fan_out = self.base_model.config.hidden_size
    
        # Xavier uniform initialization adapted for Wav2Vec 2.0
        val = math.sqrt(6. / float(fan_in + fan_out))
        self.prompts = nn.Parameter(torch.zeros(1, self.num_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)
        self.best_prompts = self.prompts

    def init_cma(self):
        """CMA-ES initialization"""
        dim = self.prompts.numel()
        popsize = 27 # which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 0.1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

    def compute_in_domain_statistics(self):
        dataset = load_dataset(['test-clean'], 'librispeech', '/home/jiahengd/tta-suta/LibriSpeech', 1, 0.)
        accumulated_hidden_states = None
        model = self.base_model
        num_batches = 0

        for i, batch in enumerate(dataset):
            if i > 32:
                break
            num_batches += 1
            # Assuming the batch contains raw waveforms
            lens, wav, texts, files = batch
            inputs = self.processor(wav, return_tensors="pt", padding="longest")
            input_values = inputs.input_values.cuda()

            with torch.no_grad():
                # Extract final hidden states
                outputs = model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Shape: (N, B, L, D)
                last_hidden = outputs.last_hidden_state
                processed_last_hidden = last_hidden.mean(dim=1)
                processed_hidden_states = torch.cat([hidden_state.mean(dim=1) for hidden_state in hidden_states], dim=1) #shape (B, N*D)
            
            # Accumulate the sum of hidden states across batches
            if accumulated_hidden_states is None:
                accumulated_hidden_states = processed_hidden_states
            else:
                accumulated_hidden_states += processed_hidden_states
            self.encoder_source_hidden.append(processed_last_hidden.squeeze(0).cpu().numpy())
        # Concatenate all hidden states across batches
        all_hidden_states = accumulated_hidden_states / num_batches  # Shape: (B, N*D)

        # Calculate batch statistics
        batch_std, batch_mean = torch.std_mean(all_hidden_states, dim=0, unbiased=False)
        return [batch_mean, batch_std]


    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.95 * self.hist_stat + 0.05 * batch_mean

    def get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            # Use full dimension matching the model's hidden size
            return self.train_info[0][-self.prompt_dim:] - self.hist_stat

    def prompt_injection(self, features):
        """
        Inject prompts into the feature sequence.
        """
        batch_size = features.size(0)
        expanded_prompts = self.prompts.expand(batch_size, -1, -1)
        features_with_prompts = torch.cat((expanded_prompts, features), dim=1)
        return features_with_prompts

    def forward(self, input_values):
        # Ensure input_values is on the same device as the model
        input_values = input_values.to(next(self.model.parameters()).device)
        self.base_model.config.output_hidden_states = True
        # Feature extraction
        features = self.base_model.feature_extractor(input_values)
        features = features.transpose(1, 2)
        # Feature projection
        if hasattr(self.model, 'wav2vec2'):
            hidden, features = self.base_model.feature_projection(features)
        else:
            hidden = self.base_model.feature_projection(features)
        position_embeddings = self.base_model.encoder.pos_conv_embed(hidden)
        hidden = hidden + position_embeddings
      
        hidden_states = self.prompt_injection(hidden)
       
        all_hidden_states = ()
        # Transformer encoder - uses the same number of layers as the original Hubert model
        with torch.no_grad():
            for layer in self.base_model.encoder.layers:
                all_hidden_states = all_hidden_states + (hidden_states,)
                transformer_output = layer(hidden_states)
                hidden_states = transformer_output[0]
        
        final_hidden_state = self.base_model.encoder.layer_norm(hidden_states[:, self.num_prompts:, :])
        all_hidden_states = tuple(h[:, self.num_prompts:, :] for h in all_hidden_states)
        all_hidden_states = all_hidden_states + (final_hidden_state,)
        
        # Project the output to logits  
        logits = self.model.lm_head(final_hidden_state)

        return logits, final_hidden_state, all_hidden_states

class FOAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.model = Wav2Vec2WithPrompts(self.system.model, 3, self.system.processor)

    def forward_and_adapt(self, x):
        """calculating shift direction, Eqn. (8)"""
        shift_vector = self.model.get_shift_vector()

        self.model.best_loss, best_outputs, batch_means = np.inf, None, []
        final_hidden_state = None

        """Sampling from CMA-ES and evaluate the new solutions.
        Note that we also compare the current solutions with the previous best one"""
        prompts, losses = self.model.es.ask() + [self.model.best_prompts.flatten().cpu()], []
        for j, prompt in enumerate(prompts):
            self.model.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).reshape_as(self.model.prompts).cuda())
            self.model.prompts.requires_grad_(False)
            outputs, loss, batch_mean, processed_final_hidden = self.forward_and_get_loss(x, shift_vector)
            batch_means.append(batch_mean[-self.model.prompt_dim:].unsqueeze(0))
            del batch_mean

            if self.model.best_loss > loss.item():
                self.model.best_prompts = self.model.prompts
                self.model.best_loss = loss.item()
                best_outputs = outputs
                outputs = None
                final_hidden_state = processed_final_hidden
            losses.append(loss.item())
            del outputs

        """CMA-ES updates, Eqn. (6)"""
        prompts = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in prompts]
        self.model.es.tell(prompts, losses)
            
        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self.model._update_hist(batch_means)

        if final_hidden_state!=None:
            self.model.encoder_target_hidden.append(final_hidden_state.squeeze(0).cpu().numpy())
        
        return best_outputs

    def forward_and_get_loss(self, x, shift_vector):
        """
        Calculate the fitness value based on activation statistics and entropy, incorporating historical statistics and activation shifting.
        """
        outputs, final_hidden_state, all_hidden_states = self.model.forward(x)
        processed_hidden_states = torch.cat([hidden_state.mean(dim=1) for hidden_state in all_hidden_states], dim=1)
        processed_final_hidden = final_hidden_state.mean(dim=1)
        source_mean = self.model.train_info[0]
        source_std = self.model.train_info[1]
        source_mean = source_mean.cuda()
        source_std = source_std.cuda()

        # Calculate batch statistics
        criterion_mse = nn.MSELoss(reduction='none').cuda()
        batch_std, batch_mean = torch.std_mean(processed_hidden_states, dim=0, unbiased=False)
        std_mse, mean_mse = criterion_mse(batch_std, source_std), criterion_mse(batch_mean, source_mean)

        discrepancy_loss = 0.2 * (std_mse.sum() + mean_mse.sum())/32    
        predicted_ids = torch.argmax(outputs, dim=-1) 
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
        
        # Calculate entropy only for non-blank tokens
        entropys = softmax_entropy(outputs/2.5)

        entropys = entropys * non_blank.float()  # Apply mask, keeping only non-blank entropies
        entropy_loss = entropys.sum(dim=1) / non_blank.sum(dim=1).clamp(min=1)
        
        loss = discrepancy_loss + entropy_loss
        
        if shift_vector is not None:
            outputs = self.model.model.lm_head(final_hidden_state + 0.1 * shift_vector)

        return outputs, loss, batch_mean, processed_final_hidden

    def _adapt(self, wavs):
        self.system.eval()
        inputs = self.system._wav_to_model_input(wavs)
        input_values = inputs.input_values.to('cuda')
        for _ in range(self.tta_config["steps"]):
            outputs = self.forward_and_adapt(input_values)
        return outputs

    def _update(self, wavs):
        pass
        
    def run(self, wavs):
        outputs = self._adapt(wavs)
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.system.processor.batch_decode(predicted_ids)
        return list(transcription)[0]





