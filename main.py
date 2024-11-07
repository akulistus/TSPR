from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.minimal_distance import MinDistance
from utils.prepare_data import prepare_data
from utils.calc_params import calc_params, calc_prob
from utils.mda import MDA
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_filepath = './data/data.xlsx'
x_total, y_total = prepare_data(data_filepath)
NRJT = x_total[:60]
FJ = x_total[60:]
NR = NRJT[:30]
JT = NRJT[30:]

NRJTFJ_train = pd.concat([NRJT[:15], NRJT[30:45], FJ[:15]], ignore_index=True)
y = pd.concat([y_total[:15], y_total[30:45], y_total[60:75]], ignore_index=True)
NRJTFJ_test = pd.concat([NRJT[15:30], NRJT[45:], FJ[15:]], ignore_index=True)

KNN = KNearestNeighbors()
KNN.fit(NRJTFJ_train, y)
KNN_p = KNearestNeighbors(mode="proximity")
KNN_p.fit(NRJTFJ_train, y)
KNN_res = []
KNN_p_res = []
for i in range(3, 11):
    res = KNN.predict(i, NRJTFJ_test)
    res_p = KNN_p.predict(i, NRJTFJ_test)
    KNN_res.append(accuracy_score(y, res))
    KNN_p_res.append(accuracy_score(y, res_p))

    # print(f'K = {i}, KNN, OA = {accuracy_score(y, res)}')
    # print(confusion_matrix(y, res))

    # print(f'K = {i}, KNN_P, OA = {accuracy_score(y, res_p)}')
    # print(confusion_matrix(y, res_p))

fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].plot(np.linspace(3,10, num=8), KNN_res)
axes[0].set_title('Наибольшее число соседей одного класса')
axes[0].set_ylabel("Общая точность")
axes[0].set_xlabel("Число соседей")
axes[1].plot(np.linspace(3,10, num=8), KNN_p_res)
axes[1].set_title('Взвешенный способ')
axes[1].set_xlabel("Число соседей")
plt.show()

pca = PCA(n_components=2)
data_reduced_space = pca.fit_transform(x_total.values)
print(f"ГК = {pca.explained_variance_ratio_}")

# Display PCA
fig, ax = plt.subplots()
ax.scatter(data_reduced_space[:30, 0], data_reduced_space[:30, 1], c=['blue'], label="НР")
ax.scatter(data_reduced_space[30:60, 0], data_reduced_space[30:60, 1], c=['red'], label="ЖТ")
ax.scatter(data_reduced_space[60:90, 0], data_reduced_space[60:90, 1], c=['green'], label="ФЖ")
ax.set_xlabel("Y1")
ax.set_ylabel("Y2")
ax.set_title("Диаграмма рассеяния")
ax.legend()
plt.show()

# MinDistance
md_FJ = MinDistance(0, 1)
md_FJ.fit(FJ.values, NRJT.values)
pred_NRJT = md_FJ.predict(NRJT.values)
pred_FJ = md_FJ.predict(FJ.values)
NRJT_mean, NRJT_std = calc_params(pred_NRJT)
FJ_mean, FJ_std = calc_params(pred_FJ)
print("~~~~FJ/NRJT mean and std ~~~~")
print(f"{FJ_mean=} {FJ_std=}")
print(f"{NRJT_mean=} {NRJT_std=}")

md_NRJT = MinDistance(0, 1)
md_NRJT.fit(NR.values, JT.values)
pred_NR = md_NRJT.predict(NR.values)
pred_JT = md_NRJT.predict(JT.values)
NR_mean, NR_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
print("~~~~NR/JT mean and std ~~~~")
print(f"{NR_mean=} {NR_std=}")
print(f"{JT_mean=} {JT_std=}")

fig, axes = plt.subplots(1, 2, sharey=True)

prob_1 = calc_prob(pred_NRJT, np.linspace(-2.0, 0.25, num=50))
prob_2 = calc_prob(pred_FJ, np.linspace(-2.0, 0.25, num=50))
sns.histplot(pred_NRJT, bins=15, binwidth=0.15, alpha=0.5, ax=axes[0], label="НР+ЖТ")
sns.histplot(pred_FJ, bins=15, binwidth=0.15, alpha=0.5, ax=axes[0], label="ФЖ")
x_values = np.linspace(-2.0, 0.25, num=50)
sns.lineplot(x=x_values, y=prob_1, ax=axes[0], label="Плотность вероятности НР+ЖТ", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[0], label="Плотность вероятности ФЖ", c="red")
fisher = Fisher()
fisher.W = md_FJ.vector_w
threshold = fisher._find_threshold(FJ.values, NRJT.values)
print(f"НР+ЖТ и ФЖ {threshold}")
print(f"НР+ЖТ и ФЖ {md_FJ.vector_w}")
axes[0].axvline(threshold, color='red', linestyle='--', label="Порог")
axes[0].set_ylabel("Количество")
axes[0].set_xlabel("W")
axes[0].legend()


prob_1 = calc_prob(pred_NR, np.linspace(-1.0, 0.7, num=100))
prob_2 = calc_prob(pred_JT, np.linspace(-1.0, 0.7, num=100))
sns.histplot(pred_NR, bins=15, binwidth=0.15, alpha=0.5, ax=axes[1], label="НР")
sns.histplot(pred_JT, bins=15, binwidth=0.15, alpha=0.5, ax=axes[1], label="ЖТ")
x_values = np.linspace(-1.0, 0.7, num=100)
sns.lineplot(x=x_values, y=prob_1, ax=axes[1], label="Плотность вероятности НР", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[1], label="Плотность вероятности ЖТ", c="red")
fisher = Fisher()
fisher.W = md_NRJT.vector_w
threshold = fisher._find_threshold(NR.values, JT.values)
print(f"НР и ЖТ {threshold}")
print(f"НР и ЖТ {md_NRJT.vector_w}")
axes[1].axvline(threshold, color='red', linestyle='--', label="Порог")
axes[1].set_xlabel("W")
axes[1].legend()

fig.suptitle("Проекции множества классов на весовой вектор W")
plt.show()


# Fisher
fisher_FJ = Fisher()
fisher_FJ.fit(FJ.values, NRJT.values)
pred_NRJT = fisher_FJ.predict(NRJT.values)
pred_FJ = fisher_FJ.predict(FJ.values)
NRJT_mean, NRJT_std = calc_params(pred_NRJT)
FJ_mean, FJ_std = calc_params(pred_FJ)
print("~~~~FJ/NRJT mean and std ~~~~")
# print(f"{FJ_mean=} {FJ_std=}")
# print(f"{NRJT_mean=} {NRJT_std=}")
print(f"{fisher_FJ.W=}")

fisher_NRJT = Fisher()
fisher_NRJT.fit(NR.values, JT.values)
pred_NR = fisher_NRJT.predict(NR.values)
pred_JT = fisher_NRJT.predict(JT.values)
NR_mean, NR_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
print("~~~~NR/JT mean and std ~~~~")
# print(f"{NR_mean=} {NR_std=}")
# print(f"{JT_mean=} {JT_std=}")
print(f"{fisher_NRJT.W=}")

fig, axes = plt.subplots(1, 2, sharey=True)

prob_1 = calc_prob(pred_NRJT, np.linspace(-1, 0.3, num=100))
prob_2 = calc_prob(pred_FJ, np.linspace(-1, 0.3, num=100))
sns.histplot(pred_NRJT, bins=15, binwidth=0.15, ax=axes[0], alpha=0.5, label="НР+ЖТ")
sns.histplot(pred_FJ, bins=15, binwidth=0.15, ax=axes[0], alpha=0.5, label="ФЖ")
x_values = np.linspace(-1, 0.3, num=100)
sns.lineplot(x=x_values, y=prob_1, ax=axes[0], label="Плотность вероятности НР+ЖТ", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[0], label="Плотность вероятности ФЖ", c="red")
print(f"{fisher_FJ.threshold=}")
axes[0].axvline(fisher_FJ.threshold, color='red', linestyle='--', label="Порог")
axes[0].set_ylabel("Количество")
axes[0].set_xlabel("W")
axes[0].legend()

prob_1 = calc_prob(pred_NR, np.linspace(-1, 0.3, num=100))
prob_2 = calc_prob(pred_JT, np.linspace(-1, 0.3, num=100))
sns.histplot(pred_NR, bins=15, binwidth=0.15, ax=axes[1], alpha=0.5, label="НР")
sns.histplot(pred_JT, bins=15, binwidth=0.15, ax=axes[1], alpha=0.5, label="ЖТ")
x_values = np.linspace(-1, 0.3, num=100)
sns.lineplot(x=x_values, y=prob_1, ax=axes[1], label="Плотность вероятности НР", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[1], label="Плотность вероятности ЖТ", c="red")
print(f"{fisher_NRJT.threshold=}")
axes[1].axvline(fisher_NRJT.threshold, color='red', linestyle='--', label="Порог")
axes[1].set_xlabel("W")
axes[1].legend()

fig.suptitle("Проекции множества классов на весовой вектор W")
plt.show()

# MDA
mda = MDA()
mda.fit(NR.values, JT.values, FJ.values)
x_FJ, y_FJ = mda.predict(FJ.values)
x_JT, y_JT = mda.predict(JT.values)
x_NR, y_NR = mda.predict(NR.values)
df = { 'W1': x_FJ, 'W2': y_FJ }
sns.scatterplot(data=df, x='W1', y='W2')
df = { 'W1': x_JT, 'W2': y_JT }
sns.scatterplot(data=df, x='W1', y='W2')
df = { 'W1': x_NR, 'W2': y_NR }
sns.scatterplot(data=df, x='W1', y='W2')
plt.show()
