import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../dice_images/0.jpg')

rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

detected_edges = cv.Canny(gray, 9, 150, 3)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))

close = cv.morphologyEx(detected_edges, cv.MORPH_CLOSE, kernel, iterations=2)

circles = cv.HoughCircles(close, cv.HOUGH_GRADIENT, 1.1, 20, param1=50, param2=30, minRadius=5, maxRadius=55)
print(circles)

plt.imshow(close, cmap='gray')
plt.show()

circles = circles[0, :]
print(circles)

for i in circles:
    # draw the outer circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

print(len(circles))
plt.imshow(rgb_img)
plt.show()

contours, hierarchy = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
print((hierarchy[0]))

x0, y0, w0, h0 = cv.boundingRect(contours[0])
cv.rectangle(rgb_img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 5)

plt.imshow(rgb_img)
plt.show()

dice0 = close[y0:y0+h0, x0:x0+w0]

plt.imshow(dice0, cmap='gray')
plt.show()

circles0 = cv.HoughCircles(dice0, cv.HOUGH_GRADIENT, 1.3, 20, param1=50, param2=30, minRadius=5, maxRadius=55)
print(len(circles0[0]))

cv.putText(rgb_img, f'score: {len(circles0[0])}', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
plt.imshow(rgb_img)
plt.show()
