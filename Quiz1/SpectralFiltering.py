
import numpy as np
import cv2
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt

# Load the color image
color_image = cv2.imread('quiz_img.jpeg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#Spectral Filtering
# Perform direct Fourier transform onto the grayscale image
f_transform = np.fft.fft2(gray_image)
f_shift = np.fft.fftshift(f_transform)

# Reduce by 75% the magnitude of the frequency component at the origin of the frequency spectrum
rows, cols = gray_image.shape
center_row, center_col = rows // 2, cols // 2

# Reduce the DC component (origin frequency component) by 75%
f_shift[center_row, center_col] *= 0.25

# Perform inverse Fourier transform onto the modified spectrum
f_ishift = np.fft.ifftshift(f_shift)
modified_image = np.fft.ifft2(f_ishift)
modified_image = np.abs(modified_image)

# Save the modified image
cv2.imwrite('modified_image.bmp', modified_image)

# Display the original and modified images for comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(gray_image, cmap='gray')
axs[0].set_title('Original Grayscale Image')
axs[0].axis('off')

axs[1].imshow(modified_image, cmap='gray')
axs[1].set_title('Modified Image after FFT')
axs[1].axis('off')

plt.show()

