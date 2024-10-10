import pandas as pd

data = pd.read_excel('./data/data.xlsx', sheet_name=[0, 1, 2])
print(data)