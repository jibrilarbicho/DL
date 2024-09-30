import numpy as np
from scipy.signal import convolve2d
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats("svg")
imgN = 20
image = np.random.randn(imgN, imgN)

kernelN = 7
Y, X = np.meshgrid(np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN))
kernel = np.exp(-(X**2 + Y**2) / 7)
print(kernel)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
ax[0].imshow(image)
ax[0].set_title('Image')
ax[1].imshow(kernel)
ax[1].set_title('Convolution kernel')

plt.show()
convoutput = np.zeros((imgN, imgN))
halfkr = kernelN // 2

for rowi in range(halfkr, imgN - halfkr):
    for coli in range(halfkr, imgN - halfkr):
        pieceOfImg = image[rowi - halfkr:rowi + halfkr + 1, :]
        pieceOfImg = pieceOfImg[:, coli - halfkr:coli + halfkr + 1]
        dotprod = np.sum(pieceOfImg * kernel[::-1, ::-1])
        convoutput[rowi, coli] = dotprod
convoutput2 = convolve2d(image, kernel, mode='valid')

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0, 0].imshow(image)
ax[0, 0].set_title('Image')

ax[0, 1].imshow(kernel)
ax[0, 1].set_title('Convolution kernel')

ax[1, 0].imshow(convoutput)
ax[1, 0].set_title('Manual convolution')

ax[1, 1].imshow(convoutput2)
ax[1, 1].set_title("Scipy's convolution")

plt.show()
bathtub = imageio.imread('https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

print(bathtub.shape)

fig = plt.figure(figsize=(10, 6))
plt.imshow(bathtub)
plt.show()
bathtub = np.mean(bathtub, axis=2)
bathtub = bathtub / np.max(bathtub)
VK = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])  # vertical kernel

# horizontal kernel
HK = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])
fig, ax = plt.subplots(2, 2, figsize=(16, 8))

ax[0, 0].imshow(VK)
ax[0, 0].set_title('Vertical kernel')

ax[0, 1].imshow(HK)
ax[0, 1].set_title('Horizontal kernel')

# Run convolution and show the result
convres = convolve2d(bathtub, VK, mode='same')
ax[1, 0].imshow(convres, cmap='gray', vmin=0, vmax=0.01)

convres = convolve2d(bathtub, HK, mode='same')
ax[1, 1].imshow(convres, cmap='gray', vmin=0, vmax=0.01)

plt.show()
