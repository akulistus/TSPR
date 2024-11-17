from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.minimal_distance import MinDistance
from utils.prepare_data import prepare_data
from utils.calc_params import calc_params, calc_stats
from utils.mda import MDA
from utils.plot_funcs import plot_hist
from utils.classify import classify
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_filepath = './data/data.xlsx'
x_total, y_total = prepare_data(data_filepath)
NR = x_total[:30]
JTFJ = x_total[30:]
FJ = JTFJ[30:]
JT = JTFJ[:30]

NRJTFJ_train = pd.concat([NR[:15], JT[:15], FJ[:15]], ignore_index=True)
y = pd.concat([y_total[:15], y_total[30:45], y_total[60:75]], ignore_index=True)
NRJTFJ_test = pd.concat([NR[15:], JT[15:], FJ[15:]], ignore_index=True)

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
md_NR = MinDistance(0, 1)
md_NR.fit(NR.values, JTFJ.values)
pred_NR = md_NR.predict(NR.values)
pred_JTFJ = md_NR.predict(JTFJ.values)
NR_mean, NR_std = calc_params(pred_NR)
JTFJ_mean, JTFJ_std = calc_params(pred_JTFJ)
md_1_pred = np.append(pred_NR, pred_JTFJ)
_md_1_binary_pred = np.array([1 if x > -0.14 else 0 for x in md_1_pred])
TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), _md_1_binary_pred)
# print("~~~~ Nr/JTFJ ~~~~")
# print(f"{NR_mean=} {NR_std=}")
# print(f"{JTFJ_mean=} {JTFJ_std=}")

md_JTFJ = MinDistance(0, 1)
md_JTFJ.fit(JT.values, FJ.values)
pred_JT = md_JTFJ.predict(JT.values)
pred_FJ = md_JTFJ.predict(FJ.values)
JT_mean, JT_std = calc_params(pred_JT)
FJ_mean, FJ_std = calc_params(pred_FJ)
md_2_pred = np.append(pred_JT, pred_FJ)
# print("~~~~ JT/FJ ~~~~")
# print(f"{JT_mean=} {JT_std=}")
# print(f"{FJ_mean=} {FJ_std=}")

fig, axes = plt.subplots(1, 2, sharey=True)

x_values = np.linspace(-0.5, 0.1, num=100)
_pred_NR = { "x": pred_NR, "hue": "НР" }
_pred_JTFJ = { "x": pred_JTFJ, "hue": "ЖТ+ФЖ" }
fisher = Fisher()
fisher.W = md_NR.vector_w
threshold = fisher._find_threshold(NR.values, JTFJ.values)
plot_hist(x_values, axes[0], -0.14, _pred_NR, _pred_JTFJ)
# print(f"{md_NR.vector_w=}")
# print(f"{-0.14}")
# _md_1_binary_pred = np.array([1 if x > -0.14 else 0 for x in md_1_pred])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), _md_1_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

x_values = np.linspace(-0.5, 0.2, num=100)
_pred_JT = { "x": pred_JT, "hue": "ЖТ" }
_pred_FJ = { "x": pred_FJ, "hue": "ФЖ" }
fisher = Fisher()
fisher.W = md_JTFJ.vector_w
threshold = fisher._find_threshold(JT.values, FJ.values)
plot_hist(x_values, axes[1], threshold, _pred_JT, _pred_FJ)
# print(f"{md_JTFJ.vector_w=}")
# print(f"{threshold=}")
# _md_2_binary_pred = np.array([1 if x > threshold else 0 for x in md_2_pred])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), _md_2_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

# _NRJT = np.append(pred_NR, pred_JT)
# md_2_pred = np.array([1 if x < threshold else 0 for x in _NRJT])

fig.suptitle("Проекции множества классов на весовой вектор W")
plt.show()

# Fisher
fisher_NR = Fisher()
fisher_NR.fit(NR.values, JTFJ.values)
pred_JTFJ = fisher_NR.predict(JTFJ.values)
pred_NR = fisher_NR.predict(NR.values)
JTFJ_mean, JTFJ_std = calc_params(pred_JTFJ)
NR_mean, NR_std = calc_params(pred_NR)
fish_1_pred = np.append(pred_NR, pred_JTFJ)
# print("~~~~ NR/JTFJ ~~~~")
# print(f"{NR_mean=} {NR_std=}")
# print(f"{JTFJ_mean=} {JTFJ_std=}")
# print(f"{fisher_NR.threshold}")
# print(f"{fisher_NR.W=}")
# _fish_1_binary_pred = np.array([1 if x > fisher_NR.threshold else 0 for x in fish_1_pred])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), _fish_1_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

fisher_JTFJ = Fisher()
fisher_JTFJ.fit(JT.values, FJ.values)
pred_FJ = fisher_JTFJ.predict(FJ.values)
pred_JT = fisher_JTFJ.predict(JT.values)
FJ_mean, FJ_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
fish_2_pred = np.append(pred_JT, pred_FJ)
# print("~~~~ JT/FJ ~~~~")
# print(f"{FJ_mean=} {FJ_std=}")
# print(f"{JT_mean=} {JT_std=}")
# print(f"{fisher_JTFJ.threshold=}")
# print(f"{fisher_JTFJ.W=}")
# _fish_2_binary_pred = np.array([1 if x > fisher_JTFJ.threshold else 0 for x in fish_2_pred])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), _fish_2_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

fig, axes = plt.subplots(1, 2, sharey=True)

x_values = np.linspace(-0.12, 0.03, num=100)
_pred_NR = { "x": pred_NR, "hue": "НР" }
_pred_JTFJ = { "x": pred_JTFJ, "hue": "ЖТ+ФЖ" }
plot_hist(x_values, axes[0], fisher_NR.threshold, _pred_NR, _pred_JTFJ)

# fish_1_pred = np.append(pred_FJ, pred_NRJT)
# fish_1_fpr, fish_1_tpr, t = roc_curve(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), fish_1_pred)

# fish_1_pred = np.array([1 if x < fisher_FJ.threshold else 0 for x in _FJNRjt])

x_values = np.linspace(-0.01, 0.11, num=100)
_pred_JT = { "x": pred_JT, "hue": "ЖТ" }
_pred_FJ = { "x": pred_FJ, "hue": "ФЖ" }
plot_hist(x_values, axes[1], fisher_JTFJ.threshold, _pred_JT, _pred_FJ)

# _FJNRjt = np.append(pred_NR, pred_JT)
# fish_2_pred = np.array([1 if x < fisher_NRJT.threshold else 0 for x in _FJNRjt])

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
x_values_1 = np.linspace(-0.06, 0, num=100)
y_values_1 = -2*x_values_1 + 0.07

x_values_2 = np.linspace(0, 0.05, num=100)
y_values_2 = 1.4*x_values_2 + 0.1
sns.lineplot(x=x_values_1, y=y_values_1, label="y= -2*x + 0.07", c="blue", linestyle='--')
sns.lineplot(x=x_values_2, y=y_values_2, label="y= 1.4*x + 0.1", c="orange", linestyle='--')
plt.title("Диаграмма рассеяния классов в уменьшенном пространстве признаков")
plt.legend()
plt.show()

_NR = np.hstack((x_NR.reshape(len(x_NR), 1), y_NR.reshape(len(y_NR), 1)))
_JTFJ = np.hstack((np.append(x_JT, x_FJ).reshape(60, 1), np.append(y_JT, y_FJ).reshape(60, 1)))

_FJNRjt = np.vstack((_NR, _JTFJ))
res_FJNRjt = np.array([t[1] - 1.4*t[0] - 0.1 for t in _FJNRjt])
# _res_1_binary_pred = np.array([1 if t[1] - 1.4*t[0] - 0.1 < 0 else 0 for t in _FJNRjt])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), _res_1_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

_FJ = np.hstack((x_FJ.reshape(len(x_FJ), 1), y_FJ.reshape(len(y_FJ), 1)))
_JT = np.hstack((x_JT.reshape(len(x_JT), 1), y_JT.reshape(len(y_JT), 1)))

_JTFJ = np.vstack((_JT, _FJ))
res_NRJT = np.array([t[1] + 2*t[0] - 0.07 for t in _JTFJ])
# _res_2_binary_pred = np.array([1 if t[1] + 2*t[0] - 0.07 > 0 else 0 for t in _JTFJ])
# TPR, FPR, ACC, FP, FN, TP, TN = calc_stats(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), _res_2_binary_pred)
# print(f"{TPR=}")
# print(f"{FPR=}")
# print(f"{ACC=}")
# print(f"{FP=}")
# print(f"{FN=}")
# print(f"{TP=}")
# print(f"{TN=}")

fig, axes = plt.subplots(1, 2, sharey=True)

RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), md_1_pred, ax=axes[0], alpha=0.5, name="Минимальное расстояние")
RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(60, dtype=int))), fish_1_pred, ax=axes[0], alpha=0.5, name="Фишер")
RocCurveDisplay.from_predictions(np.concatenate((np.zeros(30, dtype=int), np.ones(60, dtype=int))), res_FJNRjt, ax=axes[0], alpha=0.5, name="Дискриминантный анализ")

RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), md_2_pred, ax=axes[1], alpha=0.5, name="Минимальное расстояние")
RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), fish_2_pred, ax=axes[1], alpha=0.5, name="Фишер")
RocCurveDisplay.from_predictions(np.concatenate((np.ones(30, dtype=int), np.zeros(30, dtype=int))), res_NRJT, ax=axes[1], alpha=0.5, name="Дискриминантный анализ")
axes[0].set_ylabel("Чувствительность")
axes[1].set_ylabel("Чувствительность")
axes[0].set_xlabel("Специфичность")
axes[1].set_xlabel("Специфичность")
fig.suptitle("ROC-кривые каждого классификатора для первого и второго этапов")
plt.show()