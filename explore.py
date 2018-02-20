import pandas as pd

data = pd.read_csv('./data/train.csv')
names = list(data.columns.values)
print(names)