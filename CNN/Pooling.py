import torch
import torch.nn as nn
poolsize=3
stride=3
p2=nn.MaxPool2d(poolsize,stride)
p3=nn.MaxPool3d(poolsize,stride)
img2=torch.randn(1,1,3,3)
img3=torch.randn(1,2,3,3,3)

import torch

img2 = torch.randn(1, 1, 30, 30)
img3 = torch.randn(1, 3, 30, 30)

img2Pool2 = p2(img2)
print(f'2D image, 2D maxpool: {img2Pool2.shape}\n')

# img2Pool3 = p3(img2)
# print(f'2D image, 3D maxpool: {img2Pool3.shape}\n')

img3Pool2 = p2(img3)
print(f'3D image, 2D maxpool: {img3Pool2.shape}\n')

img3Pool3 = p3(img3)
print(f'3D image, 3D maxpool: {img3Pool3.shape}\n')
littlenet = nn.Sequential(
    nn.Conv2d(3, 10, 5, 3, 2),  # Convolution layer
    nn.ReLU(),                  # Activation function
    nn.AvgPool3d(3, 3),         # Average pool
    nn.Flatten(),               # Flatten the image to a vector
    nn.Linear(588, 1),          # Fully connected linear layer
    nn.Sigmoid()                # Output activation
)
img=torch.rand(1,3,128,128)
out=littlenet(img)
print(out.shape)