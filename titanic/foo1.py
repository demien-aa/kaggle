import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time

np.random.seed(1337)

df = pd.read_csv('data/train.csv')
print(df.head())
