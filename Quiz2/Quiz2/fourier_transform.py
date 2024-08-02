import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load a color image
color_image = cv2.imread('desk.png')
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# Perform direct Fourier transform
f_transform = np.fft.fft2(gray_image)
f_shift = np.fft.fftshift(f_transform)

# Apply a low-pass filter
rows, cols = gray_image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 30  # Radius of the low-pass filter
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Apply the mask and inverse Fourier transform
f_shift = f_shift * mask
f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the original color image, grayscale image, and filtered image
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(color_image)
plt.title('Original Color Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.title('Low-pass Filtered Image')
plt.axis('off')

plt.show()
