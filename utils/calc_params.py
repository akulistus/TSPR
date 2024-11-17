import numpy as np
from sklearn.metrics import confusion_matrix

def calc_params(X: np.ndarray):
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def calc_prob(X: np.ndarray, x: np.ndarray):
    mean, std = calc_params(X)
    return 1.0 / (std * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (x - mean)**2 / (2.0 * (std**2)))

def calc_stats(y_true: np.ndarray, y_pred: np.ndarray):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return TPR, TNR, ACC, FP, FN, TP, TN