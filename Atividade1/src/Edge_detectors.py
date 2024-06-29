import cv2 as cv
import numpy as np

img = cv.imread('../images/building.tif')
cv.imshow('img', img)

# Sobel
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)  # Horizontal
sobely = cv.Sobel(img, cv.CV_64F, 0, 1)  # Vertical
combined_sobel = cv.bitwise_or(sobelx, sobely)

# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
# cv.imshow('Combined Sobel', combined_sobel)

compare = np.concatenate((sobelx, sobely, combined_sobel), axis=1)  # Side by side comparison
cv.imshow('compare', compare)

cv.waitKey(0)
