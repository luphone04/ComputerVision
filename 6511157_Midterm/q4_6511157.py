#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:53:02 2024

@author: richard
"""

# For Question 3 and 4

import cv2
import numpy as np
import matplotlib.pyplot as plt


gray_image = cv2.imread('gray_image.bmp', cv2.IMREAD_GRAYSCALE)

#The direct Fourier Transform
f = np.fft.fft2(gray_image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Reduce the magnitude of a frequency component by 75%
rows, cols = gray_image.shape
crow, ccol = rows // 2 , cols // 2  

# Reduce the magnitude of the chosen frequency component
fshift[crow, ccol] = fshift[crow, ccol] * 0.25

# Inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
cv2.imwrite('modified_image.bmp', img_back)


# Display
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Modified Image')
plt.imshow(img_back, cmap='gray')
plt.axis('off')

plt.show()
