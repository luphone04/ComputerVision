import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('candy.jpeg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray_image.bmp', gray_image)

a, b, c = 1, 5, 7  

# Create an asymmetric 3x3 filter mask
filter_mask = np.array([[a, b, c],
                        [c, a, b],
                        [b, c, a]])

print("Filter Mask:\n", filter_mask)


correlated_image = cv2.filter2D(gray_image, -1, filter_mask)


cv2.imwrite('correlated_image.bmp', correlated_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 2)
plt.title('Correlation Result')
plt.imshow(correlated_image, cmap='gray')
plt.axis('off')
plt.show()

