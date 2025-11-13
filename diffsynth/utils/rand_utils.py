import time
import numpy as np
import random
import torch
from pytorch_lightning import seed_everything

def get_valid_seed(seed: int):
    MAX_UINT32 = 2**32 - 1
    seed = seed % (MAX_UINT32 + 1)
    return seed

def get_rand_seed(rank_id: int):
    seed = int(time.time() * 10) + rank_id
    seed = get_valid_seed(seed)
    return seed

def worker_init_fn_base(worker_id: int, rank_id: int):
    seed = int(time.time() * 1000) + rank_id * 100 + worker_id
    seed = get_valid_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def init_global_seed(rank: int, seed: int = None):
    if seed is None:
        print("No given seed!")
        seed = int(time.time()) + rank
    else:
        seed += rank
    seed = get_valid_seed(seed)
    seed_everything(seed)
