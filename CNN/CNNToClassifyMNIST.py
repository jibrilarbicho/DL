import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from torchsummary import summary
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats("svg")
data=np.loadtxt(open("../Data/mnist_train.csv", "rb"), delimiter="," ,skiprows=1)
labels=data[:,0]
data=data[:,1:]
dataNorm=data/np.max(data)
dataNorm=dataNorm.reshape(data.shape[0],1,28,28)
print(dataNorm.shape)
dataT=torch.tensor(dataNorm).float()
labelsT=torch.tensor(labels).long()
train_data,test_data,train_labels,test_labels=train_test_split(dataT,labelsT,test_size=0.1)
train_data=TensorDataset(train_data,train_labels)
test_data=TensorDataset(test_data,test_labels)
batchsize=32
train_loader=DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader=DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
print(train_loader.dataset.tensors[1].shape)
def createMNISTNet(printtoggle=False):
    class mnistnet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()
        ### convolution layers
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)  # size: np.floor((28 + 2*1 - 5) / 1) + 1 = 26 / 2 = 13 (/2 b/c maxpool)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)  # size: np.floor((13 + 2*1 - 5) / 1) + 1 = 11 / 2 = 5 (/2 b/c maxpool)

# compute the number of units in FClayer (number of outputs of conv2)
            expectSize = int(np.floor((5 + 2*0 - 1) / 1) + 1)  # fc1 layer has no padding or kernel, so expectSize = 20 * int(expectSize ** 2)

### fully-connected layer
            self.fc1 = nn.Linear(expectSize * expectSize * 20, 50)  # Adjusted the calculation based on the expected size

### output layer
            self.out = nn.Linear(50, 10)  # Corrected the declaration to nn.Linear
            self.print=printtoggle
        def forward(self, x):
            print(f'Input: {x.shape}') if self.print else None
    
    # convolution -> maxpool -> relu
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            print(f'Layer conv1/pool1: {x.shape}') if self.print else None
    
    # convolution -> maxpool -> relu
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            print(f'Layer conv2/pool2: {x.shape}') if self.print else None
    
    # reshape for linear layer
            nUnits = x.shape.numel() / x.shape[0]
            x = x.view(-1, int(nUnits))
    
            if self.print:
                print(f'Vectorize: {x.shape}')
            x = F.relu(self.fc1(x))
            if self.print:
                print(f'Layer fc1: {x.shape}')

            x = self.out(x)
            if self.print:
                print(f'Layer out: {x.shape}')

            return x
    net=mnistnet(printtoggle)
    lossfun=nn.CrossEntropyLoss()
    optimzer=torch.optim.Adam(net.parameters(),lr=0.001)
    return net,lossfun,optimzer
net,lossfun,optimzer=createMNISTNet(True)
X, y = next(iter(train_loader))
yHat=net(X)
print("   ")
print(y.shape)
print(yHat.shape)
loss=lossfun(yHat,y)
print("   ")
print("Loss:")
print(loss)
summary(net,(1,28,28))
def function2trainTheModel():
    # number of epochs
    numepochs = 10
    
    # create a new model
    net, lossfun, optimizer = createMNISTNet()
    
    # initialize losses
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []
    
    for epochi in range(numepochs):
        # loop over training data batches
        net.train()
        batchAcc = []
        batchLoss = []
        
        for X, y in train_loader:
            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat, y)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # compute accuracy
            matches = torch.argmax(yHat, axis=1) == y
            
            # booleans (false/true)
            matchesNumeric = matches.float()
            # convert to numbers (0/1)
            accuracyPct = 100 * torch.mean(matchesNumeric)
            
            # average and x100
            batchAcc.append(accuracyPct)
            batchLoss.append(loss.item())  # Store the loss for this batch
            
        # end of batch loop...
        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))
        
        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)
        
        # Evaluate on the test set
        net.eval()
        X, y = next(iter(test_loader))  # extract X, y from test dataloader
        with torch.no_grad():  # deactivate autograd
            yHat = net(X)
            # compute test accuracy
            testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
    
    # end epochs
    # function output
    return trainAcc, testAcc, losses, net
trainAcc, testAcc, losses, net=function2trainTheModel()
fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(losses,"s-")
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title('Model Loss')

ax[1].plot(trainAcc,"s-",label="Train")
ax[1].plot(test_data,"o-",label="Test")
ax[1].set_ylabel('Accuracy (%) ')
ax[1].set_xlabel('Epoch')
ax[1].set_title(f'Final Model Test Accuracy :{testAcc[-1]:.2f}%')

plt.show()


