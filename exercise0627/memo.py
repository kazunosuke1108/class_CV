import cv2
import numpy as np
img=cv2.imread("/home/ytpc2019a/code_ws/class_CV/exercise0627/results/img_disp_raw.jpg")
for i in range(10):
    print(i)
    img = cv2.bilateralFilter(img,30,20,20)
cv2.imwrite("/home/ytpc2019a/code_ws/class_CV/exercise0627/img_processing/bilateral.jpg",img)
