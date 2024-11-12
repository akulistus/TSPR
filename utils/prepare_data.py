import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(filepath):
    data = pd.read_excel(filepath, sheet_name=[0, 1, 2], skiprows=1, header=None)
    
    x_total = pd.DataFrame()
    y_total = pd.DataFrame()
    for key, value in data.items():
        x_total = pd.concat([x_total, value], ignore_index=True)
        y_total = pd.concat([y_total, pd.DataFrame(np.full((30, 1), key))], ignore_index=True)

    # scalar = MinMaxScaler()
    x_total = x_total.iloc[:, 1:].apply(lambda row: row / x_total.iloc[row.name, 0], axis=1)
    # print(x_total)

    # x_total = pd.DataFrame(scalar.fit_transform(x_total))
    return x_total, y_total
    
    
