import pandas as pd
import spicy as sp
import numpy as np

data = pd.read_csv('./data/train.csv')
names = list(data.columns.values)
print(names)
