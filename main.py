from utils.k_nearest_neighbours import KNearestNeighbors
from utils.fisher import Fisher
from utils.prepare_data import prepare_data
from sklearn.decomposition._pca import PCA
import matplotlib.pyplot as plt

data_filepath = './data/data.xlsx'
x_train, y_train, x_test, y_test, x_total, y_total = prepare_data(data_filepath)

KNN = KNearestNeighbors(3)
KNN.fit(x_train, y_train)

pca = PCA(n_components=2)
data_reduced_space = pca.fit_transform(x_total.values)

# Display PCA
plt.scatter(data_reduced_space[:30, 0], data_reduced_space[:30, 1], c=['blue'])
plt.scatter(data_reduced_space[30:60, 0], data_reduced_space[30:60, 1], c=['red'])
plt.scatter(data_reduced_space[60:90, 0], data_reduced_space[60:90, 1], c=['green'])
plt.show()