import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_fpr_tpr_in_range(y_true, y_pred, threshold_range, num_points=100):
    """
    Вычисляет FPR и TPR для порогов в заданном диапазоне.
    
    Параметры:
    - y_true: истинные метки классов (0 и 1)
    - y_pred: предсказанные значения (оценки или вероятности)
    - threshold_range: кортеж (min_threshold, max_threshold) - диапазон порогов
    - num_points: количество точек для вычисления (по умолчанию 100)
    
    Возвращает:
    - thresholds: массив порогов
    - fpr_list: список значений FPR
    - tpr_list: список значений TPR
    """
    min_threshold, max_threshold = threshold_range
    thresholds = np.linspace(min_threshold, max_threshold, num_points)
    fpr_list = []
    tpr_list = []

    for thr in thresholds:
        # Бинаризация предсказаний по текущему порогу
        y_pred_binary = (y_pred >= thr).astype(int)
        
        # Вычисление метрик через матрицу ошибок
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return thresholds, fpr_list, tpr_list