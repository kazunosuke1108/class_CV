# BRIEF
# https://docs.opencv.org/3.4/dc/d7d/tutorial_py_brief.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('graf/img5.ppm', 0)  # Import image
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img, None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
img2 = cv.drawKeypoints(img, kp, None, color=(
    255, 0, 0))  # reflect points detected
print(brief.descriptorSize())
print(des.shape)
plt.imshow(img2,), plt.show()
cv.imwrite('result5.jpg', img2)  # export image
