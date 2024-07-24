import numpy as np
import torch
nv1=np.array([1,2,3,4])
nv2=np.array([0,1,0,-1])
print(np.dot(nv1,nv2))
print(np.sum(nv1*nv2))

tv1=torch.tensor([1,2,3,4])
tv2=torch.tensor([0,1,0,-1])
print(torch.dot(tv1,tv2))
print(torch.sum(tv1*tv2))
