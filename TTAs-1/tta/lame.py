import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy

# Reuse affinity and optimization from original LAME implementation
def kNN_affinity(X: torch.Tensor, knn: int):
    N = X.size(0)
    # pairwise L2 distances
    dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
    k = min(knn + 1, N)
    # exclude self (distance zero)
    knn_idx = dist.topk(k, dim=-1, largest=False).indices[:, 1:]
    W = torch.zeros(N, N, device=X.device)
    W.scatter_(-1, knn_idx, 1.0)
    return 0.5 * (W + W.t())  # ensure symmetry


def laplacian_optimization(unary: torch.Tensor,
                            kernel: torch.Tensor,
                            bound_lambda: float = 1.0,
                            max_steps: int = 100,
                            tol: float = 1e-8) -> torch.Tensor:
    """
    Solve:
      min_{Y (N×C)} \sum_i KL(Y_i || q_i) - \lambda \sum_{ij} W_{ij} Y_i^T Y_j
    with CCCP closed-form updates.
    unary: [N, C] = -log(q_i(k))
    kernel: [N, N] affinity matrix W
    returns Y: [N, C] refined probabilities
    """
    N, C = unary.shape
    # initialize Y = softmax(-unary) = q_i
    Y = (-unary).softmax(dim=-1)
    old_energy = float('inf')
    for i in range(max_steps):
        # pairwise term: W @ Y yields [N, C]
        pairwise = bound_lambda * kernel.matmul(Y)
        # update numerator exponent
        exp_term = -unary + pairwise
        Y = exp_term.softmax(dim=-1)
        # compute energy to check convergence
        kl = (Y * (unary + torch.log(Y.clamp(min=1e-20)))).sum()
        smooth = -(bound_lambda * pairwise * Y).sum()
        energy = kl + smooth
        if i > 1 and (abs(old_energy - energy) <= tol * abs(old_energy)):
            break
        old_energy = energy
    return Y


class LAMEStrategy(IStrategy):
    """
    LAME adaptation for Wav2Vec2ForCTC on a single utterance (batch_size=1),
    treating frames as samples for test-time adaptation via LAME.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()

        if 'hubert' in config["system_config"]["model_name"].lower():
            self.base_model = self.system.model.hubert
        else:
            self.base_model = self.system.model.wav2vec2

        self.classifier = self.system.model.lm_head
        self.knn = self.tta_config["knn"]
        # we will use original bias for final logits if needed
        self.bias = self.system.model.lm_head.bias.clone()
        self.bound_lambda = self.tta_config["bound_lambda"]
    
    def forward(self, input_values, attention_mask=None, adapt=True):
        # 1) Feature extraction (no grad)
        with torch.no_grad():
            outputs = self.base_model(input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  
        # 2) If not adapting, return original logits
        logits = self.classifier(hidden) 
        if not adapt:
            return logits
        # 3) Prepare frame-level features and probabilities
        H = hidden.squeeze(0)          
        feats = F.normalize(H, p=2, dim=-1)
        prob = logits.softmax(dim=-1).squeeze(0)  
        
        # 4) Identify likely blank tokens
        blank_prob = prob[:, 0]  # Probability of blank token (index 0)
        blank_threshold = 0.5  # Threshold to consider a frame as blank
        likely_blank = blank_prob > blank_threshold
        
        # Save original blank probabilities
        original_blank_prob = blank_prob.clone()
        
        # 5) Create affinity matrix excluding likely blanks
        if (~likely_blank).sum() > self.knn + 1:  # Make sure we have enough non-blank frames
            # Use only non-blank frames for affinity calculation
            non_blank_feats = feats[~likely_blank]
            non_blank_idx = torch.where(~likely_blank)[0]
            
            # Calculate k-NN only on non-blank frames
            non_blank_W = kNN_affinity(non_blank_feats, self.knn)
            
            # Create full affinity matrix
            W = torch.zeros(feats.size(0), feats.size(0), device=feats.device)
            
            # Copy non-blank affinities to the right positions
            for i, idx_i in enumerate(non_blank_idx):
                for j, idx_j in enumerate(non_blank_idx):
                    W[idx_i, idx_j] = non_blank_W[i, j]
        else:
            # Not enough non-blank frames, fall back to using all frames
            W = kNN_affinity(feats, self.knn)
        
        # 6) Compute unary term from original probabilities
        unary = -torch.log(prob + 1e-10) 
        
        # 7) Run LAME optimization
        Y = laplacian_optimization(unary, W, bound_lambda=self.bound_lambda)
        
        # 8) Optional: Restore original blank probabilities where likely blank
        if True: 
            # Blend original and adapted blank probabilities
            blend_factor = 1.0 
            Y[likely_blank, 0] = blend_factor * original_blank_prob[likely_blank] + \
                                (1 - blend_factor) * Y[likely_blank, 0]
            
            # Renormalize
            Y = Y / Y.sum(dim=1, keepdim=True)
        
        # 9) Convert back to logits and return
        adapted_logits = torch.log(Y.clamp(min=1e-8))
        adapted_logits = adapted_logits.unsqueeze(0)
        
        return adapted_logits

    def _adapt(self, wavs):
        self.system.eval()
        inputs = self.system._wav_to_model_input(wavs)
        input_values = inputs.input_values.to('cuda')
        for _ in range(self.tta_config["steps"]):
            outputs = self.forward(input_values=input_values)
        return outputs
    
    def run(self, wavs):
        outputs = self._adapt(wavs)
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.system.processor.batch_decode(predicted_ids)
        return list(transcription)[0]
            
    def reset(self):
        # no internal state to reset
        pass
