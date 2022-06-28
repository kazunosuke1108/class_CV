from pprint import pprint
from typing import Type
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
        if m.distance < 0.5*n.distance:  # 厳選を実施
            good.append(m)
            good2.append([m])
    img1_pt = [list(map(int, kp1[m.queryIdx].pt))
               for m in good]  # マッチした１枚目の特徴点
    img2_pt = [list(map(int, kp2[m.trainIdx].pt))
               for m in good]  # マッチした２枚目の特徴点
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good2,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # マッチ状況の描画
    img3=cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
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


os.chdir("class_CV/exercise0627")
current_dir = os.getcwd()
# camera(current_dir+"/images",0)
# img1=cv2.imread(current_dir+"/images/img1.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/images/img2.jpg",cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/einstein/einstein1.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/einstein/einstein2.jpg",cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(current_dir+"/sosokan/IMG_8573.jpg")#, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(current_dir+"/sosokan/IMG_8574.jpg")#, cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/fukinuke/IMG_8577.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/fukinuke/IMG_8579.jpg",cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/tana/IMG_8592.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/tana/IMG_8593.jpg",cv2.IMREAD_GRAYSCALE)
# img1=cv2.imread(current_dir+"/book/IMG_8594.jpg",cv2.IMREAD_GRAYSCALE)
# img2=cv2.imread(current_dir+"/book/IMG_8595.jpg",cv2.IMREAD_GRAYSCALE)

img1_pt_s, img2_pt_s = match_feature(img1, img2)
F, mask = cv2.findFundamentalMat(img1_pt_s, img2_pt_s, cv2.RANSAC, 3, 0.99)
print("F matrix\n", F)

# K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]]) # 広角カメラ
K = np.array([[2.93374279e+03, 0.00000000e+00, 1.96523556e+03], [0.00000000e+00, 2.94658602e+03,
             1.43472275e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # 通常カメラ
E = K.T.dot(F).dot(K)  # essential matrix
print("K matrix\n", K)

retval, H1, H2 = cv2.stereoRectifyUncalibrated(
    img1_pt_s, img2_pt_s, F, img1.shape[:2])  # H matrix for rectification
print("H1 matrix\n", H1)
print("H2 matrix\n", H2)
# rectify the original image
img1r = cv2.warpPerspective(img1, H1, [img1.shape[1], img1.shape[0]])
img2r = cv2.warpPerspective(img2, H2, [img2.shape[1], img2.shape[0]])
cv2.imwrite(current_dir+"/results/img1r.jpg", img1r)
cv2.imwrite(current_dir+"/results/img2r.jpg", img2r)

# セミグローバルブロックマッチングアルゴリズムによるインスタンス化
stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=9)
# rectificationしたあとの画像から、disparity画像を生成
disparity_raw = stereo.compute(img1r, img2r)
cv2.imwrite(current_dir+"/results/img_disp_raw.jpg", disparity_raw)
disparity = (disparity_raw-disparity_raw.min()) / \
    (disparity_raw.max()-disparity_raw.min())*256  # 正規化（0から255へ）
kernel = np.ones((10, 10), np.float32)/100  # ノイズ除去のため平均化フィルタ実装
disparity = cv2.filter2D(disparity, -1, kernel)
cv2.GaussianBlur(disparity, (299, 299), 0)  # ノイズ除去のためガウシアンフィルタ実装
kernel = np.ones((10, 10), np.float32)/100  # ノイズ除去のため平均化フィルタ実装
disparity = cv2.filter2D(disparity, -1, kernel)
cv2.GaussianBlur(disparity, (299, 299), 0)  # ノイズ除去のためガウシアンフィルタ実装
cv2.imwrite(current_dir+"/results/img_disp.jpg", disparity)


def interpolate(i, imgL, imgR, disparity_raw):
    img_interp = np.zeros_like(imgL)  # 描画のキャンバスを用意
    for y in range(imgL.shape[0]):  # 画像の縦方向
        for x1 in range(imgL.shape[1]):  # 画像の横方向
            if y % 100 == 0 and x1 % 100 == 0:
                print("current process: ", y, x1)
            x2 = int(x1 - disparity_raw[y, x1])  # 左側画像に対応する点の右側座標を取得
            x_i = int((2 - i) * x1 + (i - 1) * x2)  # 左側画像と右側座標の間の位置を決定
            # 移動先が画像のサイズを超えていないか確認
            if 0 <= x_i < img_interp.shape[1] and 0 <= x2 < imgR.shape[1]:
                for col in range (3):
                    img_interp[y, x_i,col] = int(
                        (2 - i) * imgL[y, x1,col] + (i - 1) * imgR[y, x2,col])  # 　移動先の画素値を決定
                    if img_interp[y, x_i,col]==0:# 画素値が入っていなかった場合
                        try:
                            img_interp[y, x_i,col]=np.average(img_interp[y-1, x_i,col]+img_interp[y, x_i-1,col],img_interp[y-1, x_i-1,col])# 
                        except TypeError:
                            pass
                        except np.AxisError:
                            pass
    return img_interp.astype(np.uint8)


"""
for i in range(1, 10):
    deg = 1+i/10  # 左を１，右を２としたときのバーチャルカメラの位置
    # view interpolationを実行
"""
img_interp = interpolate(1.5, img1r, img2r, disparity_raw)
cv2.imwrite(current_dir+f"/results/view_interpolation/img_interp1.5_denoise.jpg", img_interp)


# import disparity_interpolation


# def interpolate_disparity(disparity_map: np.array) -> np.array:
#     """Intepolate disparity image to inpaint holes.
#        The expected run time for a stereo image with 2056 x 2464 pixels is ~50 ms.
#     """
#     # Set the invalid disparity values defined as "0" to -1.
#     disparity_map[disparity_map == 0] = -1
#     disparity_map_interp = disparity_interpolation.disparity_interpolator(disparity_map)
#     return disparity_map_interp

# interpolated=interpolate_disparity(np.float32(disparity))

# difference=interpolated-disparity
# cv2.imwrite(current_dir+"/results/diff_inter_disp.jpg",difference)
# cv2.imwrite(current_dir+"/results/interpolated.jpg",interpolated)


"""
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("", size = 20)
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")
Xmat=np.matlib.repmat(np.linspace(0,img1.shape[1],img1.shape[1]),img1.shape[0],1)
Ymat=np.matlib.repmat(np.linspace(0,img1.shape[0],img1.shape[0]),img1.shape[1],1).T

X=[]
Y=[]
Z=[]
z_ref=0
for i,row in enumerate(disparity):
    for j,pixel in enumerate(row):
        if i%50==0 and j%50==0:
            X.append(j)
            Y.append(i)
            if int(pixel)>100:
                Z.append(pixel)
                z_ref=pixel
            else:
                Z.append(np.nan)
        
X=np.array(X)
Y=np.array(Y)
Z=np.array(Z)

print(X.shape,Y.shape,Z.shape)
ax.scatter(X,Y,Z,c=Z,cmap="Greys")

plt.show()"""
