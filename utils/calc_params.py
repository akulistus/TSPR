import numpy as np

def calc_params(X: np.ndarray):
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def calc_prob(X: np.ndarray, x: np.ndarray):
    mean, std = calc_params(X)
    return 1.0 / (std * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (x - mean)**2 / (2.0 * (std**2)))