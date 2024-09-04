import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display
display.set_matplotlib_formats("svg")
import seaborn as sns
iris = sns.load_dataset('iris')
dataset=torch.tensor(iris[iris.columns[0:4]].values).float()
labels=torch.zeros(len(dataset),dtype=torch.long)
labels[iris.species=="versicolor"]=1
labels[iris.species=="virginica"]=2
ANNiris = nn.Sequential(
    nn.Linear(4, 64),  
    nn.ReLU(),        
    nn.Linear(64, 64),  
    nn.ReLU(),        
    nn.Linear(64, 3)    
)

lossfun = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)

numepochs = 1000
losses = torch.zeros(numepochs)
ongoingAcc=[]

for epochi in range(numepochs):

  yHat = ANNiris(dataset)

  loss = lossfun(yHat,labels)
  losses[epochi] = loss

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  matches = torch.argmax(yHat, axis=1) == labels
  matchesNumeric = matches.float()                
  accuracyPct = 100 * torch.mean(matchesNumeric)  
  ongoingAcc.append(accuracyPct)                  

predictions = ANNiris(dataset)
predlabels = torch.argmax(predictions, axis=1)
totalacc = 100 * torch.mean((predlabels == labels).float())  

print('Final accuracy: %g%%' % totalacc)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_title('Accuracy')

plt.show()


sm=nn.Softmax(1)


fig = plt.figure(figsize=(10, 4))
plt.plot(sm(yHat.detach()), 's-', markerfacecolor='w')
plt.xlabel('Stimulus number')
plt.ylabel('Probability')
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.show()
