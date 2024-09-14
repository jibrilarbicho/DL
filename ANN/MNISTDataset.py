import numpy as np
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats("svg")
data=np.loadtxt(open("../Data/mnist_train.csv", "rb"), delimiter="," ,skiprows=1)
labels=data[:,0]
data=data[:,1:]
# print(labels)
# print(data)
fig,axs=plt.subplots(3,4 , figsize=(10,6))
for ax in axs.flatten():
    randimg2show=np.random.randint(0, high=data.shape[0])
    img=np.reshape(data[randimg2show,:],(28,28))
    ax.imshow(img, cmap="gray")
    ax.set_title("The number %i "%labels[randimg2show])
plt.suptitle("How human see the data",fontsize=20)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
fig,axs=plt.subplots(3,4 , figsize=(10,6))
for ax in axs.flatten():
    randimg2show=np.random.randint(0, high=data.shape[0])
    img=ax.plot(data[randimg2show,:],'ko')
    ax.set_title("The number %i "%labels[randimg2show])
plt.suptitle("How FFN see the data",fontsize=20)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
the7s = np.where(labels == 7)[0]
sevens=data[the7s,:]
# Draw the first 12
fig, axs = plt.subplots(2, 6, figsize=(15, 6))
for ax in axs.flatten():
    randimg2show = np.random.randint(0, high=sevens.shape[0])
    img = np.reshape(sevens[randimg2show, :], (28, 28))
    ax.imshow(img, cmap="gray")
    ax.set_title("The number 7")
plt.suptitle("Example 7's", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()