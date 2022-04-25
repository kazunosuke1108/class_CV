import numpy as np
import cv2

img=cv2.imread("exercise0425/IMG_7678.jpg")
height, width, _ = img.shape
"""
for point in [[1560,1728],[2304,1500],[1616, 2760],[2400,2808]]:
    cv2.circle(img, tuple(point), 10, (255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
"""
cv2.imshow('input', img)
cv2.waitKey(0)

pts1 = np.float32([[1560,1728],[2304,1500],[1616, 2760],[2400,2808]])
pts2 = np.float32([[500,500],[2500,500],[500,3500],[2500,3500]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (3000, 4000))

cv2.imshow('output',dst)
cv2.waitKey(0)

cv2.destroyAllWindows()