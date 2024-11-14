import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.calc_params import calc_prob
def plot_hist(x_linespace, axe, threshold, pred_1, pred_2):
    prob_1 = calc_prob(pred_1["x"], x_linespace)
    prob_2 = calc_prob(pred_2["x"], x_linespace)
    sns.histplot(data=pred_1["x"], bins=15, alpha=0.5, ax=axe, label=pred_1["hue"])
    sns.histplot(data=pred_2["x"], bins=15, alpha=0.5, ax=axe, label=pred_2["hue"])
    sns.lineplot(x=x_linespace, y=prob_1, ax=axe, label="Плотность вероятности НР+ЖТ", c="blue")
    sns.lineplot(x=x_linespace, y=prob_2, ax=axe, label="Плотность вероятности ФЖ", c="red")
    # print(f"НР+ЖТ и ФЖ {threshold}")
    # print(f"НР+ЖТ и ФЖ {md_FJ.vector_w}")
    axe.axvline(threshold, color='red', linestyle='--', label="Порог")
    axe.set_ylabel("Количество")
    axe.set_xlabel("W")
    axe.legend()