from pprint import pprint
import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import os
from camera import camera
from mpl_toolkits.mplot3d import Axes3D

def match_feature(img1, img2):
    # SIFTを用意
    sift = cv2.SIFT_create()
    # 特徴点とdescriberを検出
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # マッチさせる
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []  # オブジェクトの保管場所
    good2 = []  # オブジェクトの保管場所 drawMatchesKnnに食わせるための形式
    for m, n in matches:
        if m.distance < 0.7*n.distance:  # 厳選を実施
            good.append(m)
            good2.append([m])
    img1_pt = [list(map(int, kp1[m.queryIdx].pt))
               for m in good]  # マッチした１枚目の特徴点
    img2_pt = [list(map(int, kp2[m.trainIdx].pt))
               for m in good]  # マッチした２枚目の特徴点
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good2,
                             None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # マッチ状況の描画
    plt.imshow(img3), plt.show()  # 確認のためマッチの様子を出力

    img1_pt_s = []  # 重複削除後のマッチした特徴点
    img2_pt_s = []
    for i in range(len(img1_pt)):
        if (img1_pt[i] not in img1_pt_s) and (img2_pt[i] not in img2_pt_s):  # 重複の確認
            img1_pt_s.append(img1_pt[i])
            img2_pt_s.append(img2_pt[i])

    img1_pt_s = np.array(img1_pt_s)  # 最適化に備えnumpy形式に変換
    img2_pt_s = np.array(img2_pt_s)
    return img1_pt_s, img2_pt_s

os.chdir("exercise0627")
current_dir=os.getcwd()
# camera(current_dir+"/images",0)
# img1=cv2.imread(current_dir+"/images/img1.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/images/img2.jpg",cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/einstein/einstein1.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/einstein/einstein2.jpg",cv2.IMREAD_GRAYSCALE)
img1=cv2.imread(current_dir+"/sosokan/IMG_8573.jpg",cv2.IMREAD_GRAYSCALE)
img2=cv2.imread(current_dir+"/sosokan/IMG_8574.jpg",cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/fukinuke/IMG_8577.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/fukinuke/IMG_8579.jpg",cv2.IMREAD_GRAYSCALE)

img1_pt_s, img2_pt_s = match_feature(img1, img2)
F, mask = cv2.findFundamentalMat(img1_pt_s, img2_pt_s, cv2.RANSAC, 3, 0.99)
print("F matrix\n",F)

# K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]]) # 広角カメラ
K=np.array([[2.93374279e+03, 0.00000000e+00, 1.96523556e+03],[0.00000000e+00, 2.94658602e+03, 1.43472275e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) # 通常カメラ
E = K.T.dot(F).dot(K)
print("K matrix\n",K)

retval, H1, H2 = cv2.stereoRectifyUncalibrated(img1_pt_s, img2_pt_s, F, img1.shape[:2])
print("H1 matrix\n",H1)
print("H2 matrix\n",H2)
img1r = cv2.warpPerspective(img1, H1, [img1.shape[1],img1.shape[0]])
img2r = cv2.warpPerspective(img2, H2, [img2.shape[1],img2.shape[0]])
cv2.imwrite(current_dir+"/results/img1r.jpg",img1r)
cv2.imwrite(current_dir+"/results/img2r.jpg",img2r)
stereo = cv2.StereoBM_create(numDisparities=256, blockSize=9)
disparity = stereo.compute(img1r,img2r)
pprint(disparity)
print(disparity.max(),disparity.min(),disparity)
disparity=(disparity-disparity.min())/(disparity.max()-disparity.min())*256

cv2.imwrite(current_dir+"/results/img_disp.jpg", disparity)
pprint(disparity)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("", size = 20)
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")
Xmat=np.matlib.repmat(np.linspace(0,img1.shape[1],img1.shape[1]),img1.shape[0],1)
Ymat=np.matlib.repmat(np.linspace(0,img1.shape[0],img1.shape[0]),img1.shape[1],1).T

z=disparity
print(Xmat.shape,Ymat.shape,z.shape)
ax.plot(Xmat.reshape(1,-1), Ymat.reshape(1,-1), z.reshape(1,-1))

plt.show()