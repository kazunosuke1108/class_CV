import cv2

img=cv2.imread("/home/ytpc2019a/code_ws/class_CV/exercise0627/results/view_interpolation/img_interp1.1.jpg")

cv2.GaussianBlur(img,(101,101), sigmaX=3)
cv2.imshow("frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()