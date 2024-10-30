from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.minimal_distance import MinDistance
from utils.prepare_data import prepare_data
from utils.calc_params import calc_params, calc_prob
from utils.mda import MDA
from sklearn.decomposition._pca import PCA
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

KNN = KNearestNeighbors(3)
KNN.fit(NRJTFJ_train, y)
KNN.predict(NRJTFJ_test)

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
