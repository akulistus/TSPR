import numpy as np
import pandas as pd

class KNearestNeighbors: 
    def __init__(self, n_neighbors, mode="distance") -> None:
        self.n_neighbors = n_neighbors
        self.mode = mode
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.x_train = x_train
        self.y_train = y_train
    
    def _euclidean_distances(self, X, x_test_i) -> None:
        return np.sqrt(np.sum((X - x_test_i) ** 2, axis=1))
    
    def _make_prediction_distance(self, x_test_i):
        distance = self._euclidean_distances(self.x_train, x_test_i)
        k_nearest_indexes = np.argsort(distance)[:self.n_neighbors]
        targets = self.y_train.iloc[[*k_nearest_indexes]]
        return np.bincount(targets.values.ravel()).argmax()
    
    def _make_prediction_proximity(self, x_test_i):
        distance = self._euclidean_distances(self.x_train, x_test_i)
        k_nearest_indexes = np.argsort(distance)[:self.n_neighbors + 1]

        # Filter the nearest neighbors based on class
        nearest_neighbors = self.x_train.iloc[k_nearest_indexes]
        nearest_labels = self.y_train.iloc[k_nearest_indexes, 0]

        # Split the DataFrame by class
        cls_1 = nearest_neighbors[nearest_labels == 0]
        cls_2 = nearest_neighbors[nearest_labels == 1]
        cls_3 = nearest_neighbors[nearest_labels == 2]
    
        proximity_1 = self._calc_proximity(pd.DataFrame(cls_1), x_test_i)
        proximity_2 = self._calc_proximity(pd.DataFrame(cls_2), x_test_i)
        proximity_3 = self._calc_proximity(pd.DataFrame(cls_3), x_test_i)

        max_proximity = max(proximity_1, proximity_2, proximity_3)
        if max_proximity == proximity_1:
            return 0
        elif max_proximity == proximity_2:
            return 1
        else:
            return 2

    
    def _calc_proximity(self, cls, x_test_i):
        return np.sum(1/np.sqrt(self._euclidean_distances(cls, x_test_i)))

    def predict(self, X_test: pd.DataFrame):
        if (self.mode == "distance"):
            return np.array([self._make_prediction_distance(row) for index, row in X_test.iterrows()])
        elif (self.mode == "proximity"):
            return np.array([self._make_prediction_proximity(row) for index, row in X_test.iterrows()])
            
    