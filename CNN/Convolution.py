import numpy as np
from scipy.signal import convolve2d
from imageio import imread
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats("svg")
imgN = 20
image = np.random.randn(imgN, imgN)

kernelN = 7
Y, X = np.meshgrid(np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN))
kernel = np.exp(-(X**2 + Y**2) / 7)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
ax[0].imshow(image)
ax[0].set_title('Image')
ax[1].imshow(kernel)
ax[1].set_title('Convolution kernel')

plt.show()
