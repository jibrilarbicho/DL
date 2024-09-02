import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
     


nPerClust = 100
blur = 1

A = [  1, 3 ]
B = [  1, -2]

a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

data_np = np.hstack((a,b)).T

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

def createANNmodel(learningRate):
    ANNclassify = nn.Sequential(
        nn.Linear(2, 16),   # Input layer
        nn.ReLU(),          # Activation unit
        nn.Linear(16, 1),   # Hidden layer
        nn.ReLU(),          # Activation unit
        nn.Linear(1, 1),    # Output unit
        nn.Sigmoid()        # Final activation unit
    )
    
    lossfun = nn.BCELoss()
    optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=learningRate)

    return ANNclassify, lossfun,optimizer
numepochs=1000
def trainTheModel(AnnModel):
    losses = torch.zeros(numepochs)
    for epochi in range(numepochs):
        yHat = AnnModel(data)
        loss = lossfun(yHat, labels)
        losses[epochi] = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predictions = AnnModel(data)
    totalacc=torch.mean(((predictions>0.5)==labels).float())
    return losses, totalacc,predictions
ANNclassify, lossfun,optimizer=createANNmodel(0.1)
losses, totalacc,predictions=trainTheModel(ANNclassify)
print('final Accuracy: %g%%' %totalacc)
plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()