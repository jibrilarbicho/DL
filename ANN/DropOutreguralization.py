import torch 
import torch.nn as nn
import torch.nn.functional as F
prob=0.25
dropout=nn.Dropout(p=prob)
x=torch.ones(10)
y=dropout(x)
print("X",x)
print("Y",y)
dropout.eval()
y=dropout(x)
print("Y",y)
#F.dropout() is not deactivated in eval mode
dropout.eval()
y=F.dropout(x)
print("Y",y)
dropout.eval()
y=F.dropout(x,training=False)# but you manullay switch it off  by including training=False 
print("Y",y)
#The models need to be reset after toggling into eval mode
dropout.train()
y=dropout(x)
print("Y",y)