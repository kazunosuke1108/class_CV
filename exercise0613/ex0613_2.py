import numpy as np
import cv2
import matplotlib.pyplot as plt

def matching(img1,img2):
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

    return pts1,pts2,F,mask

def drawlines(img1,img2,pts1,pts2):
    r,c=img1.shape
    lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, 3)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img3 = cv2.line(img1, (x0, y0), (x1, y1), color,
                        thickness=3)  # epilineを引く
    return img3

# 画像の読み込み
left_path="exercise0606/images/ex0606left10.jpg"
right_path="exercise0606/images/ex0606right10.jpg"

image_left=cv2.imread(left_path)
image_left=cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
image_right=cv2.imread(right_path)
image_right=cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

# 特徴点抽出など
pts1,pts2,F,mask=matching(image_left,image_right)
print(F)
image_left_before_rect=drawlines(image_left,image_right,pts1,pts2)
image_right_before_rect=drawlines(image_right,image_left,pts2,pts1)
cv2.imwrite("test_l.png",image_left_before_rect)
cv2.imwrite("test_r.png",image_right_before_rect)

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
for r, pt1, pt2 in zip(lines1, pts1, pts2):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -r[2]/r[1]])
    x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
    img3 = cv2.line(img1, (x0, y0), (x1, y1), color,
                    thickness=3)  # epilineを引く
for r, pt1, pt2 in zip(lines2, pts2, pts1):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -r[2]/r[1]])
    x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
    img5 = cv2.line(img2, (x0, y0), (x1, y1), color,
                    thickness=3)  # epilineを引く
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.savefig(f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/class_CV/exercise0613/results/test{num}.png")
plt.close()


# rectification前のepipolar lineを描画