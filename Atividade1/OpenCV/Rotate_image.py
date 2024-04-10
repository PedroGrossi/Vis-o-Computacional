import cv2 as cv
import numpy as np


# Rotation function
def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    if rot_point is None:
        rot_point = (width//2, height//2)

    rot_mat = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rot_mat, dimensions)


img = cv.imread('../images/3.jpg')
# cv.imshow('img', img)

rotated_img = rotate(img, 45)
# cv.imshow('rotated_img', rotated_img)

compare = np.concatenate((img, rotated_img), axis=1)  # Side by side comparison
cv.imshow('compare', compare)

cv.waitKey(0)
