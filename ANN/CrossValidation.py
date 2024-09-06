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
propTraining=.8
nTarining=int(len(labels)*propTraining)
traintestBool=np.zeros(len(labels),dtype=bool)
traintestBool[range(nTarining)]=True
import torch
import torch.nn as nn

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

  yHat = ANNiris(dataset[traintestBool,:])
  ongoingAcc.append(100*torch.mean(
(torch.argmax(yHat, axis=1) == labels [traintestBool]).float()))

  loss = lossfun(yHat,labels[traintestBool])
  losses[epochi] = loss

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  predictions = ANNiris(dataset[traintestBool, :])
trainacc = 100 * torch.mean((torch.argmax(predictions, axis=1) == labels[traintestBool]).float())

predictions = ANNiris(dataset[~traintestBool, :])
testacc = 100 * torch.mean((torch.argmax(predictions, axis=1) == labels[~traintestBool]).float())

print(f'Final TRAIN accuracy: {trainacc:.2f}%')
print(f'Final TEST accuracy: {testacc:.2f}%')
