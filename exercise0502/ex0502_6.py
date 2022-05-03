# ORB + matching
# https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
# nearest neighboring matchingではない
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# process the 1st image
img1 = cv.imread('graf/img5.ppm', 0)  # Import image
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1, None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
# draw only keypoints location,not size and orientation
img11 = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
plt.imshow(img11), plt.show()

# process the 2nd image
img2 = cv.imread('graf/img6.ppm', 0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp2 = orb.detect(img2, None)
# compute the descriptors with ORB
kp2, des2 = orb.compute(img2, kp2)
# draw only keypoints location,not size and orientation
img22 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
plt.imshow(img22), plt.show()

# matching
"""
あるfeatureのdescriptorを他方の画像のfeaturesの全てと比較して、最も近いものと結びつける
"""
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img11, kp1, img22, kp2,
                      matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
cv.imwrite('result6_m.jpg', img3)
