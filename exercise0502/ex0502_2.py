# SIFT
# https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

import numpy as np
import cv2 as cv
img = cv.imread('graf/img5.ppm')  # Import image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert color
sift = cv.SIFT_create()  # initialize
kp = sift.detect(gray, None)  # find feature points
img = cv.drawKeypoints(gray, kp, img)  # reflect points detected
cv.imwrite('result2_1.jpg', img)  # export results
# make with another function
img = cv.drawKeypoints(
    gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('result2_2.jpg', img)  # export results
