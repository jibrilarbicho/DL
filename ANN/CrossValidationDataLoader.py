import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils
from torch.utils.data import DataLoader,TensorDataset
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
# train_data, test_data, train_labels, test_labels = train_test_split(fakedata, fakelabels,train_size=0.8)

# train_data = torch.utils.data.TensorDataset(
#     torch.Tensor(train_data),
#     torch.Tensor(train_labels)
# )
# test_data = torch.utils.data.TensorDataset(
#     torch.Tensor(test_data),
#     torch.Tensor(test_labels)
# )

# train_loader = DataLoader(train_data, batch_size=4)
# test_loader = DataLoader(test_data)
# print('TRAINING DATA')
# for batch, label in train_loader:  # iterable
#     print(batch, label)
#     print('')

# print('TESTING DATA')
# for batch, label in test_loader:  # iterable
#     print(batch, label)
#     print('')

train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, train_size=0.8)

train_data = torch.utils.data.TensorDataset(train_data, train_labels)
test_data = torch.utils.data.TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_data, shuffle=True, batch_size=12)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
# for X,y in train_loader:
#     print(X,y)
#     print('')


def createANNmodel():
    ANNiris = nn.Sequential(
        nn.Linear(4, 64),   # Input layer
        nn.ReLU(),          # Activation unit
        nn.Linear(64, 64),   # Hidden layer
        nn.ReLU(),          # Activation unit
        nn.Linear(64, 3),    # Output unit
    )
    
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.1)

    return ANNiris, lossfun,optimizer
numepochs=1000
def trainTheModel():
    trainAcc=[]
    testAcc=[]
    losses = torch.zeros(numepochs)
    for epochi in range(numepochs):
        batchAcc=[]
        for X,y in train_loader:

            yHat = ANNiris(X)
            loss = lossfun(yHat, y)
            losses[epochi] = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()).item())
        trainAcc.append(np.mean(batchAcc))
        X, y = next(iter(test_loader))
        predlabels = torch.argmax(ANNiris(X), axis=1)
        testAcc.append(100 * torch.mean((predlabels == y).float()).item())
    return trainAcc, testAcc

            
   
ANNiris, lossfun,optimizer=createANNmodel()
trainAcc, testAcc=trainTheModel()
fig = plt.figure(figsize=(10, 5))
plt.plot(trainAcc, 'ro-')
plt.plot(testAcc, 'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train', 'Test'])
plt.show()

