from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.minimal_distance import MinDistance
from utils.prepare_data import prepare_data
from utils.calc_params import calc_params, calc_prob
from utils.mda import MDA
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
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

    print(f'K = {i}, KNN, OA = {accuracy_score(y, res)}')
    print(confusion_matrix(y, res))

    print(f'K = {i}, KNN_P, OA = {accuracy_score(y, res_p)}')
    print(confusion_matrix(y, res_p))

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

# Display PCA
plt.scatter(data_reduced_space[:30, 0], data_reduced_space[:30, 1], c=['blue'])
plt.scatter(data_reduced_space[30:60, 0], data_reduced_space[30:60, 1], c=['red'])
plt.scatter(data_reduced_space[60:90, 0], data_reduced_space[60:90, 1], c=['green'])
plt.show()

# MinDistance
md = MinDistance(0, 1)
md.fit(FJ.values, NRJT.values)
pred_NRJT = md.predict(NRJT.values)
pred_FJ = md.predict(FJ.values)
NRJT_mean, NRJT_std = calc_params(pred_NRJT)
FJ_mean, FJ_std = calc_params(pred_FJ)
df = { 'NRJT':md.predict(NRJT.values), 'FJ': md.predict(FJ.values)}
prob_1 = calc_prob(pred_NRJT, np.linspace(-2.0, 0.25, num=50))
prob_2 = calc_prob(pred_FJ, np.linspace(-2.0, 0.25, num=50))
ax = sns.histplot(data=df, bins=15)
ax.plot(np.linspace(-2.0, 0.25, num=50), prob_1)
ax.plot(np.linspace(-2.0, 0.25, num=50), prob_2)
ax.axvline(md.threshold)
plt.show()

# Fisher
fisher = Fisher()
fisher.fit(FJ.values, NRJT.values)
pred_NRJT = fisher.predict(NRJT.values)
pred_FJ = fisher.predict(FJ.values)
NRJT_mean, NRJT_std = calc_params(pred_NRJT)
FJ_mean, FJ_std = calc_params(pred_FJ)
df = { 'NRJT':pred_NRJT, 'FJ': pred_FJ}
prob_1 = calc_prob(pred_NRJT, np.linspace(-1, 0.3, num=100))
prob_2 = calc_prob(pred_FJ, np.linspace(-1, 0.3, num=100))
ax = sns.histplot(data=df, bins=15)
ax.plot(np.linspace(-1, 0.3, num=100), prob_1)
ax.plot(np.linspace(-1, 0.3, num=100), prob_2)
ax.axvline(fisher.threshold)
plt.show()

fisher = Fisher()
fisher.fit(NR.values, JT.values)
pred_NR = fisher.predict(NR.values)
pred_JT = fisher.predict(JT.values)
NR_mean, NR_std = calc_params(pred_NR)
JT_mean, JT_std = calc_params(pred_JT)
df = { 'NR':pred_NR, 'JT': pred_JT}
prob_1 = calc_prob(pred_NR, np.linspace(-1, 0.3, num=100))
prob_2 = calc_prob(pred_JT, np.linspace(-1, 0.3, num=100))
ax = sns.histplot(data=df, bins=15)
ax.plot(np.linspace(-1, 0.3, num=100), prob_1)
ax.plot(np.linspace(-1, 0.3, num=100), prob_2)
ax.axvline(fisher.threshold)
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
