import os 
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch import nn
from jiwer import wer
import math
import numpy as np
import random
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
from data import load_dataset

class EATAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        config["system_config"]["train_feature"] = False
        #Main model is set with AdamW optimizer
        self.system = GradientBasedSystem(config["system_config"])

        #SGD optimizer used for fisher information calculation
        config["system_config"]["optimizer"] = self.tta_config["optimizer"]
        config["system_config"]["lr"] = 0.001
        self.subnetSystem = GradientBasedSystem(config["system_config"])
        self.subnetSystem.eval()
        self.system.eval()
        self.fisher_alpha = self.tta_config["fisher_alpha"]
        self.e_margin = 0.4*math.log(32)
        self.d_margin = self.tta_config["d_margin"]
        self.model_probs = None
        self.nums_update_1 = 0
        self.nums_update_2 = 0
        self.fisher_size = self.tta_config["fisher_size"]
        self.fisher_matrix = self.compute_fisher_information()

    def _init_start(self) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, wavs):
        self.system.eval()
        for _ in range(self.tta_config["steps"]):
            record = {}
            num_counts_2, num_counts_1, updated_probs = self.system.eata_adapt(
                wavs=wavs,
                fishers=self.fisher_matrix,
                fisher_alpha=self.fisher_alpha,
                e_margin=self.e_margin,
                d_margin=self.d_margin,
                current_model_probs=self.model_probs,
                num_samples_update=self.nums_update_2,
            )
            self.nums_update_2 += num_counts_2
            self.nums_update_1 += num_counts_1
            self.model_probs = updated_probs
    
    def _update(self, wavs):
        pass
    
    def run(self, wavs):
        self._adapt(wavs)
        self.system.eval()
        trans = self.system.inference(wavs)[0]
        self._update(wavs)
        return trans

    def compute_fisher_information(self):
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        dataset = load_dataset(['test-other'], 'librispeech', '/home/jiahengd/tta-suta/LibriSpeech', 1, 0.0)
        for i, batch in enumerate(dataset):
            if i >= self.fisher_size:
                break

            # Unpack the batch
            lens, wavs, texts, files = batch

            # Prepare the input values
            inputs = self.subnetSystem.processor(wavs, return_tensors="pt", padding="longest")
            input_values = inputs.input_values.cuda()

            # Generate the model's predictions
            logits = self.subnetSystem.model(input_values).logits

            # Generate targets
            predicted_ids = torch.argmax(logits, dim=-1)

            # Simulate the loss calculation
            targets = predicted_ids.view(-1).cuda()  # Assuming the predicted IDs as the "true" targets for Fisher information
            loss = train_loss_fn(logits.view(-1, logits.size(-1)), targets)

            # Backward pass
            loss.backward()

            # Compute the Fisher information
            for name, param in self.subnetSystem.model.named_parameters():
                if param.grad is not None:
                    fisher = param.grad.data.clone().detach() ** 2
                    if name in fishers:
                        fishers[name][0] += fisher
                    else:
                        fishers[name] = [fisher, param.data.clone().detach()]
                
            self.subnetSystem.optimizer.zero_grad()

        # Normalize by the number of samples used
        for name in fishers:
            fishers[name][0] /= self.fisher_size

        return fishers