import pandas as pd
import numpy as np

def prepare_data(filepath):
    data = pd.read_excel(filepath, sheet_name=[0, 1, 2], skiprows=1, header=None)
    max_specter = np.max([value[0].max() for key, value in data.items()])
    
    x_test = pd.DataFrame()
    y_test = pd.DataFrame()
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()
    x_total = pd.DataFrame()
    y_total = pd.DataFrame()
    for key, value in data.items():
        x_train = pd.concat([x_train, value[:15] / max_specter], ignore_index=True)
        y_train = pd.concat([y_train, pd.DataFrame(np.full((15, 1), key))], ignore_index=True)

        x_test = pd.concat([x_test, value[15:] / max_specter], ignore_index=True)
        y_test = pd.concat([y_test, pd.DataFrame(np.full((15, 1), key))], ignore_index=True)

        x_total = pd.concat([x_total, value / max_specter], ignore_index=True)
        y_total = pd.concat([y_total, pd.DataFrame(np.full((30, 1), key))], ignore_index=True)

    return x_train, y_train, x_test, y_test, x_total, y_total
    
    
