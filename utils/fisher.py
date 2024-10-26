import numpy as np

class Fisher():
    def __init__(self) -> None:
        self.W = None
    
    def fit(self, class_1:np.ndarray, class_2:np.ndarray):        
        diff_mean = np.mean(class_1, axis=0) - np.mean(class_2, axis=0)
        sum_covariance = np.cov(class_1, rowvar=0) + np.cov(class_2, rowvar=0)
        if sum_covariance.ndim < 1:
            sum_covariance = np.array([[sum_covariance]])
        self.W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
        self.W = self.W/np.linalg.norm(self.W)

        self.threshold = self._find_threshold(class_1, class_2)

        return self
    
    def predict(self, data:np.ndarray):
        proj = np.matmul(data, self.W)

        return proj
        # return np.where(proj > self.threshold, 1, 0)
    
    def _find_threshold(self, class_1:np.ndarray, class_2:np.ndarray):
        proj1 = np.matmul(class_1, self.W)
        proj2 = np.matmul(class_2, self.W)

        proj1_mean = np.mean(proj1)
        proj2_mean = np.mean(proj2)

        proj1_std = np.std(proj1)
        proj2_std = np.std(proj2)

        threshold = self._solve_pdfs(proj1_mean, proj2_mean, proj1_std, proj2_std)
        return threshold

    def _solve_pdfs(self, pdf_mean_1, pdf_mean_2, pdf_std_1, pdf_std_2):
        coeff1 = 1/(2*pdf_std_1**2) - 1/(2*pdf_std_2**2)
        coeff2 = pdf_mean_2/(pdf_std_2**2) - pdf_mean_1/(pdf_std_1**2)
        coeff3 = pdf_mean_1**2 /(2*pdf_std_1**2) - pdf_mean_2**2 / (2*pdf_std_2**2) - np.log(pdf_std_2/pdf_std_1)

        coeffs = [coeff1, coeff2, coeff3]

        roots_of_eq = np.roots(coeffs)
        threshold = roots_of_eq[1]
        return threshold
