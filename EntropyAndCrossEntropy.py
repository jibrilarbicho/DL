import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
p=0.25
X=[.25,.75]
H=0
for p in X:
    H+=-p*np.log(p)
print("Entropy: ",str(H))
#Binary Cross Entropy
K=-(p*np.log(p)+(1-p)*np.log(1-p))
print(K)
p=[1,0]
q=[.25,.75]
p_tensor=torch.Tensor(p)
q_tensor=torch.Tensor(q)
print(F.binary_cross_entropy(q_tensor,p_tensor))


