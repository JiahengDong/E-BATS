import numpy as np
import torch
import random
import os

def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)

def batchify(data, batch_size, shuffle=False):
    """
    Batch generator for list data.
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_data = [data[idx] for idx in batch_idx]
        yield batch_data
