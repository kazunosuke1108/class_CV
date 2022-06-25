import glob
import os
import cv2
import numpy as np
from scipy.optimize import least_squares

#データセットの指定
data_num=1

# pathの定義
current_dir=os.getcwd()
data_dir="big_data_NO_GIT/SfM_datas"
images_dir=data_dir+f"/data{data_num}/images/observatory"
calibration_file_dir = data_dir + f'/data{data_num}/calibration'


# K行列 (カメラ行列3x3)を求める
K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]])
# 各2枚の画像について

## 特徴量マッチングをする
images=glob.glob(images_dir+"/*.JPG")
for i in range(len(images)-1):
    img1=cv2.imread(images[i],0)
    img2=cv2.imread(images[i+1],0)
    sift = cv2.SIFT_create()  # インスタンス化
    # 特徴点の抽出
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 以下、特徴点のマッチング
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 条件の良い特徴点の組みだけを残す
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    #  F行列を求める
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3, 0.99)
    src_pts = src_pts.flatten()
    dst_pts = dst_pts.flatten()
    print(F)
## F行列を求める
## E=K^-1・F・Kにより、E行列を求める  <---ここの計算が理解できていない。こっち？https://tutorialmore.com/questions-1705590.htm
## カメラの回転R、移動t、その推定の誤差retvalを計算する。

