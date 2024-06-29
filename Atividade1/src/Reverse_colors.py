import cv2 as cv
import numpy as np

img1 = cv.imread('../images/1.jpg')
img2 = cv.imread('../images/2.jpg')
# cv.imshow('img1', img1)
# cv.imshow('img2', img2)

# Invert Thresholding
threshold1, thresh_inv1 = cv.threshold(img1, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('img1_inv', thresh_inv1)
threshold2, thresh_inv2 = cv.threshold(img2, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('img2_inv', thresh_inv2)

compare1 = np.concatenate((img1, thresh_inv1), axis=1)  # Side by side comparison
cv.imshow('compare1', compare1)
compare2 = np.concatenate((img2, thresh_inv2), axis=1)  # Side by side comparison
cv.imshow('compare2', compare2)

cv.waitKey(0)
