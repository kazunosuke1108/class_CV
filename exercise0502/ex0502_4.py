# FAST
# https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('graf/img5.ppm', 0)  # Import image
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(
    255, 0, 0))  # reflect points detected
# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
cv.imwrite('result4_1.jpg', img2)  # export results
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)  # find feature points
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(
    255, 0, 0))  # reflect points detected
cv.imwrite('result4_2.jpg', img3)  # export results
