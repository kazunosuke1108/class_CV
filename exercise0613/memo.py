import numpy as np
import cv2
import matplotlib.pyplot as plt

image_left=cv2.imread("exercise0613/results/left.png")
image_right=cv2.imread("exercise0613/results/right.png")

# image_left=cv2.imread("exercise0613/images/Tsukuba_L.png")
# image_right=cv2.imread("exercise0613/images/Tsukuba_R.png")
image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
plt.imshow(image_right,'gray')

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(image_left,image_right)
plt.imshow(disparity,'gray')
plt.show()