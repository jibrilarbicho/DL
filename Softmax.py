import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
z=[1,2,3]
# num=np.exp(z)
# den=np.sum(np.exp(z))
# sigma=num/den
# print(sigma)
# print(np.sum(sigma))
# z = np.random.randint(-5, high=15, size=25)
# print(z)

# Compute the softmax result
# num = np.exp(z)
# den = np.sum(num)
# sigma = num / den

# Compare with plot
# plt.plot(z, sigma, 'ko')
# plt.xlabel('Original number (z)')
# plt.ylabel('Softmaxified $\sigma$')
# # plt.yscale('log')
# plt.title('$\sum\sigma$ = %g' % np.sum(sigma))
# plt.show()

softfun=nn.Softmax(dim=0)
sigmaT=softfun(torch.Tensor(z))
print(sigmaT)