import numpy as np
import cv2
import matplotlib.pyplot as plt

left_path="exercise0613/images/ex0606left.jpg"
right_path="exercise0613/images/ex0606right.jpg"

img1 = cv2.imread(left_path, 0)
img2 = cv2.imread(right_path, 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
# inlier pointsだけを使用
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

print(f"Fundamental Matrix of picture\n", F)
