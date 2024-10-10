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
nPerClass = 1000
imgSize = 91

x = np.linspace(-4, 4, imgSize)
X, Y = np.meshgrid(x, x)



# The two widths (a.u.)
widths = [1.8, 2.4]

# Initialize tensors containing images and labels
images = torch.zeros(2 * nPerClass, 1, imgSize, imgSize)
labels = torch.zeros(2 * nPerClass)

for i in range(2 * nPerClass):
    # Create the gaussian with random centers
    ro = 2 * np.random.randn(2)  # ro = random offset
    G = np.exp(-((X - ro[0])**2 + (Y - ro[1])**2) / (2 * widths[i % 2]**2))

# Add noise
    G = G + np.random.randn(imgSize, imgSize) / 5

# Add to the tensor
    images[i, :, :, :] = torch.Tensor(G).view(1, imgSize, imgSize)

# Assign label
    labels[i] = i % 2

# Reshape labels to have an additional dimension
    labels = labels[:, None]
fig, axs = plt.subplots(3, 7, figsize=(13, 6))

for i, ax in enumerate(axs.flatten()):
    # Randomly select an image
    whichpic = np.random.randint(2 * nPerClass)
    
    # Remove the channel dimension from the image
    G = np.squeeze(images[whichpic, :, :, :])
    
    # Display the image
    ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
    
    # Set the title with the class label
    ax.set_title('Class %s' % int(labels[whichpic].item()))
    
    # Remove x and y ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

train_data,test_data,train_labels,test_labels=train_test_split(images,labels,test_size=0.1)
train_data=TensorDataset(train_data,train_labels)
test_data=TensorDataset(test_data,test_labels)
batchsize=32
train_loader=DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader=DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
def makeTheNet():
    class GausNet(nn.Module):
        def __init__(self):
            super().__init__()
            # All layers in one go using nn.Sequential
            self.enc = nn.Sequential(
                nn.Conv2d(1, 6, 3, padding=1),   # Conv layer, input channels = 1, output channels = 6, kernel size = 3
                # Output size: (91 + 2*1 - 3) / 1 + 1 = 91
                nn.ReLU(),                       # Activation function (ReLU)
                nn.AvgPool2d(2, 2),              # Average pooling, kernel size = 2, stride = 2
                # Output size: 91 / 2 = 45
                nn.Conv2d(6, 4, 3, padding=1),   # Conv layer, input channels = 6, output channels = 4, kernel size = 3
                # Output size: (45 + 2*1 - 3) / 1 + 1 = 45
                nn.ReLU(),                       # Activation function (ReLU)
                nn.AvgPool2d(2, 2),              # Average pooling, kernel size = 2, stride = 2
                # Output size: 45 / 2 = 22
                nn.Flatten(),                    # Flatten the tensor for the fully connected layer
                nn.Linear(22*22*4, 50),          # Fully connected layer, input size = 22*22*4, output size = 50
                nn.Linear(50, 1)                 # Final fully connected layer, output size = 1
            )
        def forward(self, x):
            return self.enc(x)
    net =GausNet()
    lossfun=nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    return net,lossfun,optimizer
net,lossfun,optimizer=makeTheNet()
# X, y = next(iter(train_loader))
# yHat=net(X)
# print("   ")
# print(y.shape)
# print(yHat.shape)
# loss=lossfun(yHat,y)
# print("   ")
# print("Loss:")
# print(loss)
def function2trainTheModel():
    # Number of epochs
    numepochs = 10
    
    # Create a new model, loss function, and optimizer
    net, lossfun, optimizer = makeTheNet()
    
    # Initialize losses and accuracies
    trainLoss = torch.zeros(numepochs)
    testLoss = torch.zeros(numepochs)
    trainAcc = torch.zeros(numepochs)
    testAcc = torch.zeros(numepochs)
    
    # Loop over epochs
    for epoch_i in range(numepochs):
        batchLoss = []  # to store batch losses
        batchAcc = []   # to store batch accuracies
        
        for X, y in train_loader:
            # Forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat, y)
            
            # Backpropagation
            optimizer.zero_grad()  # Clear gradients from previous step
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters
            
            # Loss from this batch
            batchLoss.append(loss.item())
            # Accuracy from this batch
            batchAcc.append(torch.mean(((yHat > 0.5) == y).float()).item())
        trainLoss[epoch_i] = np.mean(batchLoss)
        batchAcc =100*np.mean(batchAcc)
        X, y = next(iter(test_loader))  # Extract X, y from test dataloader
        with torch.no_grad():  # Deactivates autograd for test
            yHat = net(X)
            loss = lossfun(yHat, y)
    
    # Compare the following line to the training accuracy lines
            testLoss[epoch_i] = loss.item()
    
            testAcc[epoch_i] = 100 * torch.mean(((yHat > 0.5) == y).float()).item()

# End epochs
# Function output
    return trainLoss, testLoss, trainAcc, testAcc, net


fig, ax = plt.subplots(1, 2, figsize=(16, 5))

# Plot training and testing loss
ax[0].plot(trainLoss, 's-', label='Train')
ax[0].plot(testLoss, 'o-', label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss (MSE)')
ax[0].set_title('Model loss')

# Plot training and testing accuracy
ax[1].plot(trainAcc, 's-', label='Train')
ax[1].plot(testAcc, 'o-', label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')

# Show legend and display plots
ax[1].legend()
plt.show()
