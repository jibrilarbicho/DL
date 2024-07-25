import numpy as np
import torch
v=np.array([1,40,2,-3])
minval=np.max(v)
maxval=np.min(v)
print('Min,Max :  %g ,%g'%(minval,maxval))
mindx=np.argmin(v)
maxdx=np.argmax(v)
print('Min,Max indices: %g, %g'%(mindx,maxdx))
M=np.array([[0,1,10],
[20,8,5]])
print(M) ,print(' ')
minvals1=np.min(M)
minvals2=np.min(M,axis=0) #minimum of each column
minvals3=np.min(M,axis=1)#minimum of each row
print(minvals1)
print(minvals2)
print(minvals3)
mindx1=np.argmin(M)
mindx2=np.argmin(M,axis=0) #minimum of each column
mindx3=np.argmin(M,axis=1)#minimum of each row
print(mindx1)
print(mindx2)
print(mindx3)
u=torch.tensor([1,40,2,-5])
minval1=torch.min(u)
maxval1=torch.max(u)
print('Min,Max :  %g ,%g'%(minval1,maxval1))
mindx1=torch.argmin(u)
maxdx1=torch.argmax(u)
print('Min,Max indices: %g, %g'%(mindx1,maxdx1))
U=torch.tensor([[0,1,10],
[20,8,5]])
print(U) ,print(' ')
minvals1=torch.min(U)
minvals2=torch.min(U,dim=0) #minimum of each column
minvals3=torch.min(U,dim=1)#minimum of each row
print(minvals1)
print(minvals2)
print(minvals3)
