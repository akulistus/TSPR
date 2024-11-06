import numpy as np

class MinDistance:
    def __init__(self, c1_name, c2_name) -> None:
        self.c1_name = c1_name
        self.c2_name = c2_name
        self.vector_w = None
        self.threshold = None
    
    def fit(self, c1: np.ndarray, c2: np.ndarray) -> None:
        c1_mean = np.mean(c1, axis=0)
        c2_mean = np.mean(c2, axis=0)
        _w = c1_mean - c2_mean
        self.vector_w = _w/np.linalg.norm(_w)
        sumM = (c1_mean + c2_mean)
        self.threshold = np.matmul(self.vector_w.reshape((1, len(self.vector_w))), sumM.reshape((len(self.vector_w),1))* 0.5)
    
    def predict(self, X: np.ndarray):
        _proj = np.matmul(self.vector_w, X.T)
        return _proj
            