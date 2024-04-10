import cv2 as cv
import numpy as np

img = cv.imread('../images/Mona_Lisa.jpg')
# cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

inv_gray = cv.bitwise_not(gray)
# cv.imshow('inv_gray', inv_gray)

blur = cv.GaussianBlur(inv_gray, (101, 101), 0)
# cv.imshow('blur', blur)
inv_blur = cv.bitwise_not(blur)
# cv.imshow('inv_blur', inv_blur)

sketch_img = cv.divide(gray, inv_blur, scale=255.0)
# cv.imshow('sketch_img', sketch_img)

compare = np.concatenate((gray, sketch_img), axis=1)  # Side by side comparison
cv.imshow('compare', compare)

cv.waitKey(0)
