import numpy as np
import pandas as pd

class KNearestNeighbors: 
    def __init__(self, n_neighbors, regression = False) -> None:
        self.n_neighbors = n_neighbors
        self.regression = regression
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.x_train = x_train
        self.y_train = y_train
    
    def _euclidean_distances(self, x_test_i) -> None:
        return np.sqrt(np.sum((self.x_train - x_test_i) ** 2, axis=1))
    
    def _make_prediction(self, x_test_i):
        distance = self._euclidean_distances(x_test_i)
        k_nearest_indexes = np.argsort(distance)[:self.n_neighbors]
        targets = self.y_train.iloc[[*k_nearest_indexes]]
        return np.mean(targets) if self.regression else np.bincount(targets.values.ravel()).argmax()
    
    def predict(self, X_test: pd.DataFrame):
        return np.array([self._make_prediction(row) for index, row in X_test.iterrows()])
    