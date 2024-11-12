import numpy as np

class MDA:
    def __init__(self) -> None:
        return None

    def fit(self, class_1: np.ndarray, class_2: np.ndarray, class_3: np.ndarray):
        cov_1 = np.cov(class_1, rowvar=0)
        M_1 = np.mean(class_1, axis=0)
        cov_2 = np.cov(class_2, rowvar=0)
        M_2 = np.mean(class_2, axis=0)
        cov_3 = np.cov(class_3, rowvar=0)
        M_3 = np.mean(class_3, axis=0)
        M = np.mean([*class_1, *class_2, *class_3], axis=0)

        Sw = (1/3)*(cov_1 + cov_2 + cov_3)
        Sb = 0
        for i in [M_1, M_2, M_3]:
            Sb = Sb + (1/3)*np.matmul((i - M).reshape(len(class_1[0]), 1),(i - M).reshape((1, len(class_1[0]))))

        W = np.linalg.inv(Sw).dot(Sb)
        eigenvalues, eigenvectors = np.linalg.eig(W)
        idx = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvectors = eigenvectors[:, idx]  # Sort eigenvectors

        self.W_1 = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
        self.W_2 = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])

    def predict(self, cs: np.ndarray):
        pred_w1 = np.matmul(self.W_1, cs.T)
        pred_w2 = np.matmul(self.W_2, cs.T)

        return pred_w1, pred_w2



