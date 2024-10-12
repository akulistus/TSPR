from utils.k_nearest_neighbours import KNearestNeighbors
from utils.prepare_data import prepare_data
from sklearn.neighbors import KNeighborsClassifier

data_filepath = './data/data.xlsx'
x_train, y_train, x_test, y_test = prepare_data(data_filepath)

KNN = KNearestNeighbors(3)
KNN.fit(x_train, y_train)
print(KNN.predict(x_test))