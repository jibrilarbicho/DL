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
# for i in data.keys():
#     print(f'{i} Ihas {len (np.unique(data[i]))} unique values')
data = data[data['total sulfur dioxide']<200]

# fig,ax = plt.subplots(1, figsize=(17,4))
# ax = sns.boxplot(data=data)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45) 
# plt.show()
cols2zscore=data.keys()
cols2zscore=cols2zscore.drop("quality")
for col in cols2zscore:
    mean=np.mean(data[col])
    std=np.std(data[col])
    data[col]=(data[col]-mean)/std
fig,ax = plt.subplots(1, figsize=(17,4))
ax = sns.boxplot(data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45) 
plt.show()
data["boolQuality"]=0
data["boolQuality"][data["quality"]>5]=1
print(data[["quality","boolQuality"]])
dataT=torch.tensor(data[cols2zscore].values).float()
labelsT=torch.tensor(data["boolQuality"].values).float()
labelsT=labelsT[:,None]
print(dataT)
print(labelsT.shape)
train_data, test_data, train_labels, test_lables=train_test_split(dataT, labelsT, test_size=0.2)
train_data=TensorDataset(train_data, train_labels)
test_data=TensorDataset(test_data, test_lables)
batchsize=64
train_loader=DataLoader(train_data, batch_size=batchsize, shuffle=True ,drop_last=True)
test_loader=DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
for X,y in train_loader:
    print(X.shape,y.shape)
    
