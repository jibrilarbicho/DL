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
dataT=torch.tensor(dataNorm).float()
def createTheMNISTnet():
    class aeNet(nn.Module):
        def __init__(self):
            super().__init__() 
            ## input layer
            self.input=nn.Linear(784,250)
            ## hidden layer
            self.enc=nn.Linear(250,50)
            self.latent=nn.Linear(50,250)
            #output layer
            self.dec=nn.Linear(250,784)
        def forward(self, x):
            x=F.relu(self.input(x))
            x=F.relu(self.enc(x))
            x=F.relu(self.latent(x))
            return torch.sigmoid(self.dec(x))
    net=aeNet()
    lossfun=nn.NLLLoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01)
    return net, lossfun, optimizer
net, lossfun, optimizer=createTheMNISTnet()
# print(Yhat)
# print(Yhat.shape)
X=dataT[:5,:]
yHat=net(X)
fig, axs = plt.subplots(2, 5, figsize=(10, 3))
for i in range(5):
   axs[0,i].imshow(X[i,:].view(28,28).detach(),cmap='gray')
   axs[1,i].imshow(yHat[i,:].view(28,28).detach(),cmap='gray')
   axs [0,i].set_xticks([]), axs [0,i].set_yticks([])
   axs [1,i].set_xticks([]), axs [1,i].set_yticks([])
plt.suptitle('Yikes!!!')
plt.show()


def function2trainTheModel():
# number of epochs 
  numepochs = 10000 
# create a new model 
  net, lossfun, optimizer = createTheMNISTnet()
# initialize losses 
  losses = torch.zeros (numepochs)
# loop over epochs
  for epochi in range(numepochs):
# select a random set of images 
    randomidx = np.random.choice(dataT.shape[0],size=32)
    X= dataT[randomidx,:]
    yHat=net(X)
    loss=lossfun(yHat,X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[epochi]=loss.item()
  return losses,net


losses,net = function2trainTheModel()
print(f'Final loss: {losses[-1]:.4f}')
# visualize the losses
plt.plot(losses,'.-')
plt.xlabel('Epochs')
plt.ylabel('Model loss')
plt.title('OK, but what did it actually learn??')
plt.show()

