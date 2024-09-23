import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import pandas as pd
import sklearn.metrics as skm
display.set_matplotlib_formats("svg")
url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data=pd.read_csv(url,sep=";")

data = data[data['total sulfur dioxide']<200]
cols2zscore=data.keys()
cols2zscore=cols2zscore.drop("quality")
data[cols2zscore]=data[cols2zscore].apply(stats.zscore)
data["boolQuality"]=0
data["boolQuality"][data["quality"]>5]=1
print(data[["quality","boolQuality"]])
dataT=torch.tensor(data[cols2zscore].values).float()
labelsT=torch.tensor(data["boolQuality"].values).float()
labelsT=labelsT[:,None]
train_data, test_data, train_labels, test_lables=train_test_split(dataT, labelsT, test_size=0.1)
train_data=TensorDataset(train_data, train_labels)
test_data=TensorDataset(test_data, test_lables)
batchsize=64
train_loader=DataLoader(train_data, batch_size=batchsize, shuffle=True ,drop_last=True)
test_loader=DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
class AnnWine(nn.Module):
    def __init__ (self):
        super().__init__()
        self.input=nn.Linear(11,16)
        self.fc1=nn.Linear(16,32)
        self.fc2=nn.Linear(32,32)
        self.output=nn.Linear(32,1)
    def forward(self,x):
        x=F.relu(self.input(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.output(x)

        
numepochs=1000

def trainTheModel():
    lossfun=nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(winenet.parameters(), lr=0.1) 
    losses=torch.zeros(numepochs)
    trainAcc=[]
    testAcc=[]
    for epochi in range(numepochs):
        batchAcc=[]
        batchLoss=[]
        for X,y in train_loader:
            yHat=winenet(X)
            loss=lossfun(yHat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchLoss.append(loss.item())
            batchAcc.append(100 * torch.mean(((yHat > 0) == y).float()).item())

        losses[epochi]=np.mean(batchLoss)
        trainAcc.append(np.mean(batchAcc))
        X,y=next(iter(test_loader))
        with torch.no_grad():
            yHat=winenet(X)
            testAcc.append(100*torch.mean(((yHat>0)==y).float()).item())
        return trainAcc,testAcc,losses
winenet=AnnWine()
trainAcc,testAcc,losses=trainTheModel()
train_predictions=winenet(train_loader.dataset.tensors[0])
test_predictions=winenet(test_loader.dataset.tensors[0])
train_metrics = [0, 0, 0, 0]
test_metrics = [0, 0, 0, 0]

# training
train_metrics[0] = skm.accuracy_score(train_loader.dataset.tensors[1], train_predictions>0)
train_metrics[1] = skm.precision_score(train_loader.dataset.tensors[1], train_predictions>0)
train_metrics[2] = skm.recall_score(train_loader.dataset.tensors[1], train_predictions>0)
train_metrics[3] = skm.f1_score(train_loader.dataset.tensors[1], train_predictions>0)

# test
test_metrics[0] = skm.accuracy_score(test_loader.dataset.tensors[1], test_predictions > 0)
test_metrics[1] = skm.precision_score(test_loader.dataset.tensors[1], test_predictions > 0)
test_metrics[2] = skm.recall_score(test_loader.dataset.tensors[1], test_predictions > 0)
test_metrics[3] = skm.f1_score(test_loader.dataset.tensors[1], test_predictions > 0)
plt.bar(np.arange(4)-.1, train_metrics, .5)
plt.bar(np.arange(4)+.1, test_metrics, .5)
plt.xticks([0, 1, 2, 3], ['Accuracy', 'Precision', 'Recall', 'F1-score'])
plt.ylim([.6, 1])
plt.legend(['Train', 'Test'])
plt.title('Performance metrics')
plt.show()

# Confusion matrices
trainConf = skm.confusion_matrix(train_loader.dataset.tensors[1], train_predictions > 0)
testConf = skm.confusion_matrix(test_loader.dataset.tensors[1], test_predictions > 0)
print("trainconf",trainConf)
print("testConf",testConf)