import numpy as np
import torch
nv=np.array([[1,2,3,4,5]])
print(nv)
print(nv.T)
nvT=nv.T
print(nvT.T)
tv=torch.tensor([[1,2,3,4,5]])
print("torch")
print(tv)
print(tv.T)
tvT = tv.T
print(tvT.T)