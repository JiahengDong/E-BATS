import os 
import torch
from torch import nn
import numpy as np

def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
    
def renyi_entropy(x, alpha, dim=-1):
    if alpha == 1:
        return torch.mean(softmax_entropy(x, dim))
    if alpha == 'inf' or alpha == float('inf'):
        entropy, _ = torch.max(x, dim)
        return -torch.mean(torch.log(entropy))
    entropy = torch.log(torch.pow(x.softmax(dim), alpha).sum(dim)) 
    entropy = entropy / (1 - alpha)
    return torch.mean(entropy)

def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) 
    p = p.squeeze(0) 
    if reweight:
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) 
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) 
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
   
    return mcc_loss

def div_loss(x, non_blank=None, L_thd=64):
    # maximize entropy of class prediction for every time-step in a utterance 
    loss = 0
    x = x.squeeze(0)
    L = x.shape[0]

    if non_blank is not None: 
        cls_pred = x.mean(0)[1:] 
    else:
        cls_pred = x.mean(0) 

    loss = -softmax_entropy(cls_pred, 0)

    return loss

def tc_reg_loss(x, non_blank=None):
    # temporal coherence reg
    x_trans = x.transpose(1,2) 
    k = 1
    att = torch.matmul(x_trans, x)
    att = torch.softmax(att, dim=-1) 
    att_x = torch.matmul(att, x_trans) + x_trans 
    non_blank = non_blank[:, k:]
    tc_loss = att_x[:,k:][non_blank] - att_x[:, :-k][non_blank]
    tc_loss = torch.norm(tc_loss, p=2, dim=-1).mean(0)

    return tc_loss

def log_softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray = np.log(s)
    out = tmp - out
    return out