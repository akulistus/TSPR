import numpy as np

class KNearestNeighbors: 
    def __init__(self, n_neighbors, regression) -> None:
        self.n_neighbors = n_neighbors
        self.regression = regression
    
    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train
    
    def _euclidean_distances(self, x_test_i) -> None:
        return 