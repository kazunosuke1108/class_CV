# SURF 特許の関係で実装が困難なためスキップ
# https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('graf/img5.ppm',0)
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

print( surf.getHessianThreshold() )


# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)

kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

print( surf.descriptorSize() )
print(surf.getExtended())
surf.setExtended(True)
kp, des = surf.detectAndCompute(img,None)
print( surf.descriptorSize() )
print( des.shape )