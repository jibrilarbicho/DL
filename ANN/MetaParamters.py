import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data=pd.read_csv(url,sep=";")
# print(data.describe())
for i in data.keys():
    print(f'{i} Ihas {len (np.unique(data[i]))} unique values')
cols2zscore=data.keys()
cols2zscore=cols2zscore.drop("quality")
for col in cols2zscore:
    mean=np.mean(data[col])
    std=np.std(data[col])
    data[col]=(data[col]-mean)/std
print(data)

