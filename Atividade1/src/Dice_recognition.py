import cv2 as cv
import matplotlib.pyplot as plt

# Image 0

img = cv.imread('../dice_images/0.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')
# plt.show()

detected_edges = cv.Canny(gray, 9, 150, 3)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
close = cv.morphologyEx(detected_edges, cv.MORPH_CLOSE, kernel, iterations=2)
circles = cv.HoughCircles(close, cv.HOUGH_GRADIENT, 1.1, 20, param1=50, param2=30, minRadius=5, maxRadius=55)
# print(circles)
# plt.imshow(close, cmap='gray')
# plt.show()

circles = circles[0, :]
# print(circles)

for i in circles:
    # draw the outer circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

# print(len(circles))
# plt.imshow(rgb_img)
# plt.show()

contours, hierarchy = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# print((hierarchy[0]))

x0, y0, w0, h0 = cv.boundingRect(contours[0])
cv.rectangle(rgb_img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 5)
# plt.imshow(rgb_img)
# plt.show()

dice0 = close[y0:y0+h0, x0:x0+w0]
# plt.imshow(dice0, cmap='gray')
# plt.show()

circles0 = cv.HoughCircles(dice0, cv.HOUGH_GRADIENT, 1.3, 20, param1=50, param2=30, minRadius=5, maxRadius=55)
# print(len(circles0[0]))

cv.putText(rgb_img, f'score: {len(circles0[0])}', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
plt.imshow(rgb_img)
plt.show()

# Image 1

img = cv.imread('../dice_images/1.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 220, 255, cv.THRESH_BINARY_INV)[1]  # Modify threshold
detected_edges = cv.Canny(thresh, 9, 150, 3)
circles = cv.HoughCircles(detected_edges, cv.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
# print(circles)
# plt.imshow(detected_edges, cmap='gray')
# plt.show()

circles = circles[0, :]
# print(circles)

for i in circles:
    # draw the outer circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

# print(len(circles))
# plt.imshow(rgb_img)
# plt.show()

contours, hierarchy = cv.findContours(detected_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# print((hierarchy[0]))

# since we have 2 dices, and we want score of each one we need to split 'em into two image then recognise their scores.
x0, y0, w0, h0 = cv.boundingRect(contours[0])
cv.rectangle(rgb_img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 5)

x1, y1, w1, h1 = cv.boundingRect(contours[1])
cv.rectangle(rgb_img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 5)
# plt.imshow(rgb_img)
# plt.show()

dice0 = detected_edges[y0:y0+h0, x0:x0+w0]
dice1 = detected_edges[y1:y1+h1, x1:x1+w1]
# plt.imshow(dice0, cmap='gray')
# plt.show()
# plt.imshow(dice1, cmap='gray')
# plt.show()

circles0 = cv.HoughCircles(dice0, cv.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles1 = cv.HoughCircles(dice1, cv.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
# print(len(circles0[0]), len(circles1[0]))

cv.putText(rgb_img, f'score: {len(circles0[0])}', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles1[0])}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
plt.imshow(rgb_img)
plt.show()

# Image 2

img = cv.imread('../dice_images/2.png')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

detected_edges = cv.Canny(gray, 9, 150, 3)
circles = cv.HoughCircles(detected_edges, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
# print(circles)
# plt.imshow(detected_edges, cmap='gray')
# plt.show()

circles = circles[0, :]
for i in circles:
    # draw the outer circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(rgb_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

# plt.imshow(rgb_img)
# plt.show()

contours, hierarchy = cv.findContours(detected_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# print((hierarchy[0]))

# since we have 6 dices, and we want score of each one we need to split 'em into two image then recognise their scores.
x0, y0, w0, h0 = cv.boundingRect(contours[0])
cv.rectangle(rgb_img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 5)

x1, y1, w1, h1 = cv.boundingRect(contours[1])
cv.rectangle(rgb_img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 5)

x2, y2, w2, h2 = cv.boundingRect(contours[2])
cv.rectangle(rgb_img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 5)

x3, y3, w3, h3 = cv.boundingRect(contours[14])
cv.rectangle(rgb_img, (x3, y3), (x3+w3, y3+h3), (0, 255, 0), 5)

x4, y4, w4, h4 = cv.boundingRect(contours[20])
cv.rectangle(rgb_img, (x4, y4), (x4+w4, y4+h4), (0, 255, 0), 5)

x5, y5, w5, h5 = cv.boundingRect(contours[18])
cv.rectangle(rgb_img, (x5, y5), (x5+w5, y5+h5), (0, 255, 0), 5)
# plt.imshow(rgb_img)
# plt.show()

dice0 = detected_edges[y3:y3+h3, x3:x3+w3]
dice1 = detected_edges[y4:y4+h4, x4:x4+w4]
dice2 = detected_edges[y5:y5+h5, x5:x5+w5]
dice3 = detected_edges[y2:y2+h2, x2:x2+w2]
dice4 = detected_edges[y0:y0+h0, x0:x0+w0]
dice5 = detected_edges[y1:y1+h1, x1:x1+w1]

# plt.imshow(dice0, cmap='gray')
# plt.show()
# plt.imshow(dice1, cmap='gray')
# plt.show()
# plt.imshow(dice2, cmap='gray')
# plt.show()
# plt.imshow(dice3, cmap='gray')
# plt.show()
# plt.imshow(dice4, cmap='gray')
# plt.show()
# plt.imshow(dice5, cmap='gray')
# plt.show()

circles0 = cv.HoughCircles(dice0, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles1 = cv.HoughCircles(dice1, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles2 = cv.HoughCircles(dice2, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles3 = cv.HoughCircles(dice3, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles4 = cv.HoughCircles(dice4, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
circles5 = cv.HoughCircles(dice5, cv.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=25, minRadius=3, maxRadius=35)
# print(len(circles0[0]), len(circles1[0]), len(circles2[0]), len(circles3[0]), len(circles4[0]), len(circles5[0]))

cv.putText(rgb_img, f'score: {len(circles0[0])}', (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles1[0])}', (x4, y4), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles2[0])}', (x5, y5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles3[0])}', (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles4[0])}', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.putText(rgb_img, f'score: {len(circles5[0])}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
plt.imshow(rgb_img)
plt.show()
