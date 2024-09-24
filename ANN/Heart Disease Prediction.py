import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
# import the data
url  = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
data = pd.read_csv(url,sep=',',header=None)
data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','DISEASE']

# data contain some ?'s; replace with NaN and drop those rows
data = data.replace('?',np.nan).dropna()

print(data)
print(data.describe())
cols2zscore = data.keys()
cols2zscore = cols2zscore.drop(['sex', 'fbs', 'exang', 'DISEASE'])
cols2zscore

for c in cols2zscore:
    d = pd.to_numeric(data[c])  # force to numeric (addresses some data-format issues)
    data[c] = (d - d.mean()) / d.std(ddof=1)

fig, ax = plt.subplots(1, figsize=(17, 4))
ax = sns.boxplot(data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
print(data["DISEASE"].value_counts())
data["DISEASE"][data["DISEASE"]>0]=1
print(data["DISEASE"].value_counts())
dataT  = torch.tensor( data[data.keys().drop('DISEASE')].values ).float()
labels = torch.tensor( data['DISEASE'].values ).float()

print( dataT.shape )
print( labels.shape )

# we'll actually need the labels to be a "matrix"
labels = labels[:,None]
print( labels.shape )
     
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=50)
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)
batchsize    = 20
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])
     

# check sizes of data batches
for X,y in train_loader:
  print(X.shape,  y.shape)
class theNet(nn.Module):
    def __init__(self):
        super(theNet, self).__init__()
        self.input=nn.Linear(13,32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)
        self.output = nn.Linear(10, 1)
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)
net = theNet()
optimizer = torch.optim.Adam(net.parameters(),lr=.0001)
lossfun = nn.BCEWithLogitsLoss() # try with different loss function


# number of training epochs
numepochs = 100


# initialize losses and accuracies
trainLoss = torch.zeros(numepochs)
testLoss  = torch.zeros(numepochs)
trainAcc  = torch.zeros(numepochs)
testAcc   = torch.zeros(numepochs)


# loop over epochs
for epochi in range(numepochs):

  # loop over training data batches
    batchLoss = []
    for X,y in train_loader:

    # forward pass and loss
         yHat = net(X)
         loss = lossfun(yHat,y)

    # backprop
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         batchLoss.append(loss.item())
    
    # train accuracy
         predictions = (torch.sigmoid(yHat)>.5).float()
         trainAcc[epochi] = 100*torch.mean((predictions==y).float())
    trainLoss[epochi] = np.mean(batchLoss)


  ## now for the test
    X,y = next(iter(test_loader))
    yHat = net(X)
  
  # test loss
    loss = lossfun(yHat,y)
    testLoss[epochi] = loss.item()
  
  # test accuracy
    predictions = (torch.sigmoid(yHat)>.5).float()
    testAcc[epochi] = 100*torch.mean((predictions==y).float())

fig,ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(trainLoss,'s-',label='Train')
ax[0].plot(testLoss,'s-',label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model loss')

ax[1].plot(trainAcc,'s-',label='Train')
ax[1].plot(testAcc,'o-',label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
ax[1].legend()

plt.show()