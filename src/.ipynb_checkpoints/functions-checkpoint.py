import numpy as np


def mse(X, y):
    
    num_samples = y.shape[0]
    mse = (1/2) * num_samples**(-1) * np.sum((X - y)**2)
    
    return mse


def rmse(X, y):
    
    num_samples = y.shape[0]
    rmse = num_samples**(-1) * np.sum((X - y)**2)**(1/2)
    
    return rmse