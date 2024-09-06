import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import seaborn as sns
iris = sns.load_dataset('iris')
dataset=torch.tensor(iris[iris.columns[0:4]].values).float()
labels=torch.zeros(len(dataset),dtype=torch.long)
labels[iris.species=="versicolor"]=1
labels[iris.species=="virginica"]=2

fakedata = np.tile(np.array([1, 2, 3, 4]), (10, 1)) + np.tile(10 * np.arange(1, 11), (4, 1)).T
fakelabels = np.arange(10) > 4

print(fakedata)
print('')  # Print an empty line
print(fakelabels)
