import numpy as np
import torch
A=np.random.randn(3,4)
B=np.random.randn(4,5)
C=np.random.randn(3,7)
print(np.round(A@B,2)) #@  for matrix multiplication    
D=torch.randn(3,4)
E=torch.randn(4,5)
F=torch.randn(3,7)
print(np.round(D@E,2)) #@  for matrix multiplication