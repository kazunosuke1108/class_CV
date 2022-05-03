# Shi-Tomasi Corner Detector
# https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('graf/img5.ppm')  # Import image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert color
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)  # get corner
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)  # draw feature points
plt.imshow(img), plt.show()
cv.imwrite('result1.jpg', img)
