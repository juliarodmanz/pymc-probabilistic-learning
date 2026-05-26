from numpy.random import SeedSequence, default_rng
def set_global_seed(seed: int = 2003): 
    import random, os, numpy as np 
    import torch
    random.seed(seed)
    np.random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed) 
    torch.manual_seed(seed)
    try: 
        import pytensor 
        pytensor.random.seed(seed) 
    except Exception:
        pass 
    return default_rng(seed)