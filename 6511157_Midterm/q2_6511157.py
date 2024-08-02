#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:40:03 2024

@author: richard
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread('candy.jpeg')


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imwrite('gray_image.bmp', gray_image)


a, b, c = 1, 5, 7  


filter_mask = np.array([[a, b, c],
                        [c, a, b],
                        [b, c, a]])

print("Filter Mask:\n", filter_mask)


correlated_image = cv2.filter2D(gray_image, -1, filter_mask)


cv2.imwrite('correlated_image.bmp', correlated_image)

#fLippin the filter mask 
flipped_filter_mask = np.flipud(np.fliplr(filter_mask))


# Perform the convolution
convolved_image = cv2.filter2D(gray_image, -1, flipped_filter_mask)


cv2.imwrite('convolved_image.bmp', convolved_image)


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Correlation Result')
plt.imshow(correlated_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Convolution Result')
plt.imshow(convolved_image, cmap='gray')
plt.axis('off')

plt.show()
