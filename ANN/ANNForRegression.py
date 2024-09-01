import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display
display.set_matplotlib_formats("svg")
N=30
x=torch.randn(N,1)
y=x+torch.randn(N,1)/2
plt.plot(x,y, "s")
plt.show()
print(x,y)
# ANNreg=nn.Sequential(
#     nn.Linear(1,1),
#     nn.ReLU(),
#     nn.Linear(1,1)
# )
# learningrate=0.05
# lossfunction=nn.MSELoss()
# optimizer=torch.optim.SGD(ANNreg.parameters(), lr=learningrate)
# numepochs=500
# losses=torch.zeros(numepochs, dtype=torch.float)
# for epoch in range(numepochs):
#     yHat=ANNreg(x)
#     loss=lossfunction(yHat,y)
#     losses[epoch]=loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# predictions=ANNreg(x)

# testloss=(predictions-y).pow(2).mean()

# plt.plot(losses.detach(),"o", markerfacecolor="w",linewidth=.1)
# plt.plot(numepochs,testloss.detach(),"ro")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()
# plt.plot(x,y, 'bo', label='Real data')
# plt.plot(x,predictions.detach(), 'rs', label='Predictions')
# plt.title(f'prediction-data r={np.corrcoef (y.T, predictions.detach().T) [0,1]:.2f}')
# plt.legend()
# plt.show()