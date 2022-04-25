import numpy as np
import cv2

img=cv2.imread("exercise0425/IMG_7678.jpg")
height, width, _ = img.shape

cv2.imshow('input', img)
cv2.waitKey(0)

pts1 = np.float32([[343,385],[507,336],[347, 616],[527,626]])
pts2 = np.float32([[100,200],[500,200],[100,700],[500, 700]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (600, 800))

cv2.imshow('output',dst)
cv2.waitKey(0)

cv2.destroyAllWindows()