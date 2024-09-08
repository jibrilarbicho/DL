import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display
display.set_matplotlib_formats("svg")
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional  as F


iris = sns.load_dataset('iris')
dataset=torch.tensor(iris[iris.columns[0:4]].values).float()
labels=torch.zeros(len(dataset),dtype=torch.long)
labels[iris.species=="versicolor"]=1
labels[iris.species=="virginica"]=2
train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, train_size=0.8)

train_data = torch.utils.data.TensorDataset(train_data, train_labels)
test_data = torch.utils.data.TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
class theModelClass(nn.Module):
    def __init__(self,dropoutRate):
        super(theModelClass, self).__init__()
        self.input = nn.Linear(4, 12)
        self.hidden = nn.Linear(12, 12)
        self.output = nn.Linear(12, 3)
        self.dr=dropoutRate
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.dropout(x,self.dr,training=self.training)
        x = F.relu(self.hidden(x))
        x = F.dropout(x,self.dr,training=self.training)
        x = self.output(x)
        return x
def createTheModel(dropoutRate):
    ANNiris=theModelClass(dropoutRate)
    lossfun=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(ANNiris.parameters(), lr=0.005)
    return ANNiris, lossfun, optimizer
numepochs=500
def trainTheModel():
    trainAcc=[]
    testAcc=[]
    for epochi in range(numepochs):
        ANNiris.train()
        batchAcc=[]
        for X,y in train_loader:
            yHat = ANNiris(X)
            loss = lossfun(yHat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()).item())
        trainAcc.append(np.mean(batchAcc))
        ANNiris.eval()
        X, y = next(iter(test_loader))
        predlabels = torch.argmax(ANNiris(X), axis=1)
        testAcc.append(100 * torch.mean((predlabels == y).float()).item())
    return trainAcc, testAcc
dropoutrate = .0
ANNiris, lossfun, optimizer = createTheModel(dropoutrate)
trainAcc, testAcc=trainTheModel()
fig=plt.figure(figsize=(10,5))
plt.plot(trainAcc, 'ro-')
plt.plot(testAcc, 'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train', 'Test'])
plt.show()


