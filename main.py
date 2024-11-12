from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.minimal_distance import MinDistance
from utils.prepare_data import prepare_data
from utils.calc_params import calc_params, calc_prob
from utils.mda import MDA
from utils.classify import classify
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, RocCurveDisplay
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
# print("~~~~FJ/NRJT mean and std ~~~~")
# print(f"{FJ_mean=} {FJ_std=}")
# print(f"{NRJT_mean=} {NRJT_std=}")

md_NRJT = MinDistance(0, 1)
md_NRJT.fit(NR.values, JT.values)
pred_NR = md_NRJT.predict(NR.values)
pred_JT = md_NRJT.predict(JT.values)
NR_mean, NR_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
# print("~~~~NR/JT mean and std ~~~~")
# print(f"{NR_mean=} {NR_std=}")
# print(f"{JT_mean=} {JT_std=}")

fig, axes = plt.subplots(1, 2, sharey=True)

x_values = np.linspace(-0.3, 0.4, num=100)
prob_1 = calc_prob(pred_NRJT, x_values)
prob_2 = calc_prob(pred_FJ, x_values)
sns.histplot(pred_NRJT, bins=15, binwidth=0.15, alpha=0.5, ax=axes[0], label="НР+ЖТ")
sns.histplot(pred_FJ, bins=15, binwidth=0.15, alpha=0.5, ax=axes[0], label="ФЖ")
sns.lineplot(x=x_values, y=prob_1, ax=axes[0], label="Плотность вероятности НР+ЖТ", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[0], label="Плотность вероятности ФЖ", c="red")
fisher = Fisher()
fisher.W = md_FJ.vector_w
threshold = fisher._find_threshold(FJ.values, NRJT.values)
# print(f"НР+ЖТ и ФЖ {threshold}")
# print(f"НР+ЖТ и ФЖ {md_FJ.vector_w}")
axes[0].axvline(threshold, color='red', linestyle='--', label="Порог")
axes[0].set_ylabel("Количество")
axes[0].set_xlabel("W")
axes[0].legend()

md_1_pred = np.append(pred_FJ, pred_NRJT)
md_1_fpr, md_1_tpr, t = roc_curve(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), md_1_pred)

x_values = np.linspace(-0.4, 0.1, num=100)
prob_1 = calc_prob(pred_NR, x_values)
prob_2 = calc_prob(pred_JT, x_values)
sns.histplot(pred_NR, bins=15, binwidth=0.15, alpha=0.5, ax=axes[1], label="НР")
sns.histplot(pred_JT, bins=15, binwidth=0.15, alpha=0.5, ax=axes[1], label="ЖТ")
sns.lineplot(x=x_values, y=prob_1, ax=axes[1], label="Плотность вероятности НР", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[1], label="Плотность вероятности ЖТ", c="red")
fisher = Fisher()
fisher.W = md_NRJT.vector_w
threshold = fisher._find_threshold(NR.values, JT.values)
# print(f"НР и ЖТ {threshold}")
# print(f"НР и ЖТ {md_NRJT.vector_w}")
axes[1].axvline(threshold, color='red', linestyle='--', label="Порог")
axes[1].set_xlabel("W")
axes[1].legend()

_NRJT = np.append(pred_NR, pred_JT)
md_2_pred = np.array([1 if x < threshold else 0 for x in _NRJT])

fig.suptitle("Проекции множества классов на весовой вектор W")
plt.show()


# Fisher
fisher_FJ = Fisher()
fisher_FJ.fit(FJ.values, NRJT.values)
pred_NRJT = fisher_FJ.predict(NRJT.values)
pred_FJ = fisher_FJ.predict(FJ.values)
NRJT_mean, NRJT_std = calc_params(pred_NRJT)
FJ_mean, FJ_std = calc_params(pred_FJ)
# print("~~~~FJ/NRJT mean and std ~~~~")
# print(f"{FJ_mean=} {FJ_std=}")
# print(f"{NRJT_mean=} {NRJT_std=}")
# print(f"{fisher_FJ.W=}")

fisher_NRJT = Fisher()
fisher_NRJT.fit(NR.values, JT.values)
pred_NR = fisher_NRJT.predict(NR.values)
pred_JT = fisher_NRJT.predict(JT.values)
NR_mean, NR_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
# print("~~~~NR/JT mean and std ~~~~")
# print(f"{NR_mean=} {NR_std=}")
# print(f"{JT_mean=} {JT_std=}")
# print(f"{fisher_NRJT.W=}")

fig, axes = plt.subplots(1, 2, sharey=True)

x_values = np.linspace(-0.2, 0.1, num=100)
prob_1 = calc_prob(pred_NRJT, x_values)
prob_2 = calc_prob(pred_FJ, x_values)
sns.histplot(pred_NRJT, bins=15, ax=axes[0], alpha=0.5, label="НР+ЖТ")
sns.histplot(pred_FJ, bins=15, ax=axes[0], alpha=0.5, label="ФЖ")
sns.lineplot(x=x_values, y=prob_1, ax=axes[0], label="Плотность вероятности НР+ЖТ", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[0], label="Плотность вероятности ФЖ", c="red")
print(f"{fisher_FJ.threshold=}")
axes[0].axvline(fisher_FJ.threshold, color='red', linestyle='--', label="Порог")
axes[0].set_ylabel("Количество")
axes[0].set_xlabel("W")
axes[0].legend()

fish_1_pred = np.append(pred_FJ, pred_NRJT)
fish_1_fpr, fish_1_tpr, t = roc_curve(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), fish_1_pred)

# fish_1_pred = np.array([1 if x < fisher_FJ.threshold else 0 for x in _FJNRjt])

x_values = np.linspace(-0.13, 0.05, num=100)
prob_1 = calc_prob(pred_NR, x_values)
prob_2 = calc_prob(pred_JT, x_values)
sns.histplot(pred_NR, bins=15, ax=axes[1], alpha=0.5, label="НР")
sns.histplot(pred_JT, bins=15, ax=axes[1], alpha=0.5, label="ЖТ")
sns.lineplot(x=x_values, y=prob_1, ax=axes[1], label="Плотность вероятности НР", c="blue")
sns.lineplot(x=x_values, y=prob_2, ax=axes[1], label="Плотность вероятности ЖТ", c="red")
print(f"{fisher_NRJT.threshold=}")
axes[1].axvline(fisher_NRJT.threshold, color='red', linestyle='--', label="Порог")
axes[1].set_xlabel("W")
axes[1].legend()

_FJNRjt = np.append(pred_NR, pred_JT)
fish_2_pred = np.array([1 if x < fisher_NRJT.threshold else 0 for x in _FJNRjt])

fig.suptitle("Проекции множества классов на весовой вектор W")
plt.show()

# MDA
mda = MDA()
mda.fit(NR.values, JT.values, FJ.values)
x_FJ, y_FJ = mda.predict(FJ.values)
x_JT, y_JT = mda.predict(JT.values)
x_NR, y_NR = mda.predict(NR.values)
df = { 'W1': x_FJ, 'W2': y_FJ }
sns.scatterplot(data=df, x='W1', y='W2', label="ФЖ")
df = { 'W1': x_JT, 'W2': y_JT }
sns.scatterplot(data=df, x='W1', y='W2', label="ЖТ")
df = { 'W1': x_NR, 'W2': y_NR }
sns.scatterplot(data=df, x='W1', y='W2', label="НР")
x_values_1 = np.linspace(-0.8, 0.4, num=100)
y_values_1 = x_values_1 - 0.43

x_values_2 = np.linspace(-0.8, 0.4, num=100)
y_values_2 = -x_values_2 - 0.55
# sns.lineplot(x=x_values_1, y=y_values_1, label="y=x-0.43", c="blue", linestyle='--')
# sns.lineplot(x=x_values_2, y=y_values_2, label="y=-x-0.55", c="orange", linestyle='--')
plt.title("Диаграмма рассеяния классов в уменьшенном пространстве признаков")
plt.legend()
plt.show()

_FJ = np.hstack((x_FJ.reshape(len(x_FJ), 1), y_FJ.reshape(len(y_FJ), 1)))
_NRJT = np.hstack((np.append(x_NR, x_JT).reshape(60, 1), np.append(y_NR, y_JT).reshape(60, 1)))

_FJNRjt = np.vstack((_FJ, _NRJT))
res_FJNRjt = np.array([t[1] - t[0] + 0.43 for t in _FJNRjt])

_NR = np.hstack((x_NR.reshape(len(x_NR), 1), y_NR.reshape(len(y_NR), 1)))
_JT = np.hstack((x_JT.reshape(len(x_JT), 1), y_JT.reshape(len(y_JT), 1)))

_NRJT = np.vstack((_NR, _JT))
# res_NRJT = np.array([(t[1] + t[0] + 0.55, t for t in _NRJT)])

fig, axes = plt.subplots(1, 2, sharey=True)

RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), md_1_pred, ax=axes[0], alpha=0.5, linewidth=2)
RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), fish_1_pred, ax=axes[0], alpha=0.5, linewidth=2)
RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), res_FJNRjt.real, ax=axes[0], alpha=0.5)

# RocCurveDisplay.from_predictions(np.concatenate((np.zeros(30, dtype=int), np.ones(30, dtype=int))), md_2_pred, ax=axes[1])
# RocCurveDisplay.from_predictions(np.concatenate((np.zeros(30, dtype=int), np.ones(30, dtype=int))), fish_2_pred, ax=axes[1])
# RocCurveDisplay.from_predictions(np.concatenate((np.zeros(30, dtype=int), np.ones(30, dtype=int))), res_NRJT, ax=axes[1])
plt.show()