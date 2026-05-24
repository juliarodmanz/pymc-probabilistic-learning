from numpy.random import SeedSequence, default_rng
def set_global_seed(seed: int = 42): 
    import random, os, numpy as np 
    random.seed(seed)
    np.random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed) 
    try: 
        import pytensor 
        pytensor.random.seed(seed) 
    except Exception:
        pass 
    return default_rng(seed)