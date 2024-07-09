
import numpy as np
import cv2
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt

# Load the color image
color_image = cv2.imread('quiz_img.jpeg')

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('grayscale_image.bmp', gray_image)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Last three digits of admission number: 157
filter_mask = np.array([
    [1, 5, 7],
    [5, 1, 7],
    [7, 5, 1]
], dtype=np.float32)

# Normalize the filter mask
filter_mask /= np.sum(filter_mask)

print("Asymmetric Smoothing Filter Mask:")
print(filter_mask)

# Perform correlation
correlated_image = correlate2d(gray_image, filter_mask, mode='same', boundary='wrap')

# Save the correlated image as BMP
cv2.imwrite('correlated_image.bmp', correlated_image)

# Perform convolution
convolved_image = convolve2d(gray_image, filter_mask, mode='same', boundary='wrap')

# Save the convolved image as BMP
cv2.imwrite('convolved_image.bmp', convolved_image)

# Display the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(correlated_image, cmap='gray')
axs[0].set_title('Correlated Image')
axs[0].axis('off')

axs[1].imshow(convolved_image, cmap='gray')
axs[1].set_title('Convolved Image')
axs[1].axis('off')

plt.show()

