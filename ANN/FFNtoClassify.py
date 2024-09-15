import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

display.set_matplotlib_formats("svg")
data=np.loadtxt(open("../Data/mnist_train.csv", "rb"), delimiter="," ,skiprows=1)
labels=data[:,0]
data=data[:,1:]
print(data.shape)
dataNorm=data/np.max(data)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(data.flatten(), 50)
ax[0].set_xlabel('Pixel intensity values')
ax[0].set_ylabel('Count')
ax[0].set_title('Histogram of original data')
ax[0].set_yscale("log")

ax[1].hist(dataNorm.flatten(), 50)
ax[1].set_xlabel('Pixel intensity values')
ax[1].set_ylabel('Count')
ax[1].set_title('Histogram of normalized data')

plt.show()
dataT=torch.tensor(dataNorm).float()
labelsT=torch.tensor(labels).long()
train_data,test_data,train_labels,test_labels=train_test_split(dataT,labelsT,train_size=0.1)
train_data=TensorDataset(train_data,train_labels)
test_data=TensorDataset(test_data,test_labels)
batchsize=32
train_loader=DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader=DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
def createTheMNISTnet():
    class mnistNet(nn.Module):
        def __init__(self):
            super().__init__() 
            ## input layer
            self.input=nn.Linear(784,64)
            ## hidden layer
            self.fc1=nn.Linear(64,32)
            self.fc2=nn.Linear(32,32)
            #output layer
            self.output=nn.Linear(32,10)
        def forward(self, x):
            x=F.relu(self.input(x))
            x=F.relu(self.fc1(x))
            x=F.relu(self.fc2(x))
            x=self.output(x)
            return torch.log_softmax(x,axis=1)
    net=mnistNet()
    lossfun=nn.NLLLoss()
    optimizer=torch.optim.SGD(net.parameters(), lr=0.01)
    return net, lossfun, optimizer
net, lossfun, optimizer=createTheMNISTnet()
X, y = next(iter(train_loader))
Yhat=net(X)
loss=lossfun(Yhat,y)
# print(Yhat)
# print(Yhat.shape)


def function2TrainModel():
    numepochs = 60
    net, lossfun, optimizer = createTheMNISTnet()
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []

    for epochi in range(numepochs):
        batchAcc = []
        batchLoss = []
        
        # Training loop
        for X, y in train_loader:
            Yhat = net(X)
            loss = lossfun(Yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batchLoss.append(loss.item())
            
            matches = torch.argmax(Yhat, axis=1) == y
            matchesnumeric = matches.float()
            accuracyPct = 100 * torch.mean(matchesnumeric).item()
            batchAcc.append(accuracyPct)
        
        # Append average accuracy and loss for this epoch
        trainAcc.append(np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)
        
        X, y = next(iter(test_loader))
        Yhat = net(X)
        testAccuracy = 100 * torch.mean((torch.argmax(Yhat, axis=1) == y).float()).item()
        testAcc.append(testAccuracy)

    return losses, trainAcc, testAcc, net


losses, trainAcc,testAcc,net=function2TrainModel()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

# Plot loss

ax[0].plot(losses)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim([0, 3])
ax[0].set_title('Model Loss')

# Plot accuracy
ax[1].plot(trainAcc, label='Train')
ax[1].plot(testAcc, label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_ylim([10, 100])
ax[1].set_title(f'Final Model Test Accuracy: {testAcc[-1]:.2f}%')
ax[1].legend()

plt.show()
X, y = next(iter(test_loader))
predictions=net(X).detach()
# print(predictions)
sampleshow=120
# print(predictions[120])
plt.bar(range(10),torch.exp(predictions[sampleshow]))
plt.xticks(range(10))
plt.xlabel("Number")
plt.ylabel("Evidence For That Number")
plt.title('True Number was %s' %y[sampleshow].item())
plt.show()