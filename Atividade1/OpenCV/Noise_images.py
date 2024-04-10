import cv2 as cv
import numpy as np
import random


def add_noise(img):
    # Getting the dimensions of the image
    (height, width) = img.shape[:2]

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, height - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, width - 1)
        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, height - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, width - 1)
        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


img = cv.imread('../images/mrbean.jfif')
# cv.imshow('img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

salt_and_pepper_img = add_noise(gray.copy())
# cv.imshow('salt_and_pepper_img', salt_and_pepper_img)

median = cv.medianBlur(salt_and_pepper_img, 5)
# cv.imshow('median_img', median)

compare = np.concatenate((gray, salt_and_pepper_img, median), axis=1)  # Side by side comparison
cv.imshow('compare', compare)

cv.waitKey(0)
