import numpy as np
def rmse(y_true, y_pred): 
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def decompose_uncertainty(var_total, sigma2_mean): 
    epi = np.maximum(var_total - sigma2_mean, 0.0) 
    return sigma2_mean, epi