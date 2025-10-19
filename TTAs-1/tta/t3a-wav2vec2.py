import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
from systems.loss import softmax_entropy

def get_wav2vec2_featurer(model):
    """Extract the feature part of Wav2Vec2ForCTC model"""
    # Create a copy of the model for feature extraction only
    featurer = type(model)(model.config)
    featurer.load_state_dict(model.state_dict())
    
    # Remove the final projection layer that converts to vocab size
    featurer.lm_head = nn.Identity()
    return featurer

class T3AStrategy(IStrategy):
    """
    Test Time Template Adjustments (T3A) adapted for Wav2Vec2ForCTC models
    with frame-level adaptation
    """
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.classifier = self.system.model.lm_head  # The projection layer to vocab
        self.featurizer = get_wav2vec2_featurer(self.system.model)
        if 'hubert' in config["system_config"]["model_name"].lower():
            self.base_model = self.system.model.hubert
        else:
            self.base_model = self.system.model.wav2vec2
        
        # Parameters
        self.filter_K = self.tta_config["filter_K"]
        self.num_classes = self.system.model.config.vocab_size
        self.blank_index = 0  # CTC blank token index

        # Get initial model weights as supports
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        # Check normalized vectors
        normalized_supports = F.normalize(warmup_supports, dim=1)
        warmup_prob = normalized_supports @ normalized_supports.T
        
        # Compute initial entropy
        self.warmup_ent = softmax_entropy(warmup_prob, dim=-1)
        for i in range(self.num_classes):
            self.warmup_ent[i] = 1e-8

        self.warmup_labels = torch.nn.functional.one_hot(torch.argmax(warmup_prob, dim=-1), num_classes=self.num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        

    def forward(self, input_values, attention_mask=None, adapt=True):
        """
        Forward pass with optional adaptation
        
        Args:
            input_values: Input audio tensor [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            adapt: Whether to adapt the model with this batch
        
        Returns:
            logits: Output logits for CTC decoding [batch_size, seq_len, num_classes]
        """
        # Get hidden states from the model
        with torch.no_grad():
            outputs = self.base_model(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        # When not adapting, use original model directly
        if not adapt:
            with torch.no_grad():
                return self.classifier(hidden_states)
        
        # Get current predictions
        logits = self.classifier(hidden_states)
        
        # Reshape for frame-level processing
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten batch and sequence dimensions for frame-level processing
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  
        flat_logits = logits.reshape(-1, self.num_classes) 
        
        # Get frame-level predictions
        predicted_ids = torch.argmax(flat_logits, dim=-1)
        
        # Filter out blank tokens
        non_blank_mask = (predicted_ids != self.blank_index)
        
        if non_blank_mask.any():  # Only adapt if we have non-blank predictions
            # Get entropy for non-blank frames
            frame_ent = softmax_entropy(flat_logits, dim=-1)
            
            # Create one-hot pseudo-labels
            yhat = F.one_hot(predicted_ids, num_classes=self.num_classes).float()
            
            # Select only non-blank frames for adaptation
            non_blank_features = flat_hidden[non_blank_mask]  
            non_blank_labels = yhat[non_blank_mask]  
            non_blank_ent = frame_ent[non_blank_mask] 
            
            # Add to support set
            self.supports = self.supports.to(non_blank_features.device)
            self.labels = self.labels.to(non_blank_features.device)
            self.ent = self.ent.to(non_blank_features.device)

            self.supports = torch.cat([self.supports, flat_hidden])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, frame_ent])
        
        # Get filtered supports
        supports, labels = self.select_supports()
        
        # Store original weight norms for each class
        original_weights = self.classifier.weight.data
        original_norms = torch.norm(original_weights, dim=1)

        # Normalize both supports and hidden states
        supports = F.normalize(supports, dim=1)  # Normalize each support vector

        class_weights = torch.zeros(hidden_dim, self.num_classes, device=supports.device)
        for i in range(self.num_classes):
            class_mask = (labels.argmax(dim=1) == i)
            if class_mask.sum() > 0:
                class_weights[:, i] = supports[class_mask].mean(dim=0)
            else:
                class_weights[:, i] = self.classifier.weight.data[i]
                
        # Calculate weights (class prototypes)
        weights = (supports.T @ labels)

        normalized_weights = F.normalize(weights, dim=0)  # Normalize each class prototype

        # Calculate logits using normalized hidden states and scaled weights (better performance here compared with only cosine similarity)
        adapted_logits = hidden_states @ class_weights + self.classifier.bias.data
        return adapted_logits
        

    def select_supports(self):
        """Select support samples with lowest entropy for each class"""
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        
        if filter_K == -1:
            indices = torch.arange(len(ent_s), device=ent_s.device)
            return self.supports, self.labels
            
        indices = []
        all_indices = torch.arange(len(ent_s), device=ent_s.device)
        
        # Filter top-K samples per class based on entropy
        for i in range(self.num_classes):
            class_mask = (y_hat == i)
            if not class_mask.any():
                continue
                
            class_indices = all_indices[class_mask]
            class_entropy = ent_s[class_mask]
            
            # Sort by entropy (lower is better)
            _, sorted_idx = torch.sort(class_entropy)
            # Take top K
            k = min(filter_K, len(sorted_idx))
            selected = class_indices[sorted_idx[:k]]
            indices.append(selected)
            
        if indices:
            indices = torch.cat(indices)
            self.supports = self.supports[indices]
            self.labels = self.labels[indices]
            self.ent = self.ent[indices]
        
        return self.supports, self.labels
    
    def reset(self):
        """Reset adaptation to initial state"""
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
    
    def _adapt(self, wavs):
        self.system.eval()
        inputs = self.system._wav_to_model_input(wavs)
        input_values = inputs.input_values.to('cuda')
        for _ in range(self.tta_config["steps"]):
            outputs = self.forward(
                input_values=input_values,
            )
        return outputs

    def _update(self, wavs):
        pass
    
    def run(self, wavs):
        outputs = self._adapt(wavs)
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.system.processor.batch_decode(predicted_ids)
        return list(transcription)[0]
    
    