import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils
from torch.utils.data import DataLoader
import seaborn as sns
import torch.utils.data
iris = sns.load_dataset('iris')
dataset=torch.tensor(iris[iris.columns[0:4]].values).float()
labels=torch.zeros(len(dataset),dtype=torch.long)
labels[iris.species=="versicolor"]=1
labels[iris.species=="virginica"]=2

fakedata = np.tile(np.array([1, 2, 3, 4]), (10, 1)) + np.tile(10 * np.arange(1, 11), (4, 1)).T
fakelabels = np.arange(10) > 4

# print(fakedata)
# print('')  # Print an empty line
# print(fakelabels)
# fakeDataLoader=DataLoader(fakedata,shuffle=True)
# print(fakeDataLoader)
# print(fakeDataLoader.batch_size)
# for i, oneSample in enumerate(fakeDataLoader):
#     print(i,oneSample)
# fakeDatset=torch.utils.data.TensorDataset(torch.Tensor(fakedata),torch.Tensor(fakelabels))
# fakeDataLoader=DataLoader(fakeDatset,shuffle=True)

# for data, lab in enumerate(fakeDataLoader):
#     print(data,lab)
train_data, test_data, train_labels, test_labels = train_test_split(fakedata, fakelabels,train_size=0.8)

train_data = torch.utils.data.TensorDataset(
    torch.Tensor(train_data),
    torch.Tensor(train_labels)
)
test_data = torch.utils.data.TensorDataset(
    torch.Tensor(test_data),
    torch.Tensor(test_labels)
)

train_loader = DataLoader(train_data, batch_size=4)
test_loader = DataLoader(test_data)
print('TRAINING DATA')
for batch, label in train_loader:  # iterable
    print(batch, label)
    print('')

print('TESTING DATA')
for batch, label in test_loader:  # iterable
    print(batch, label)
    print('')

