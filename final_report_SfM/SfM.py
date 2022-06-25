import glob
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from mayavi import mlab


def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


def viz_3d(pt_3d):
    X = pt_3d[0,:]
    Y = pt_3d[1,:]
    Z = pt_3d[2,:]

    mlab.points3d(
        X,   # x
        Y,   # y
        Z,   # z
        mode="point", # How to render each point {'point', 'sphere' , 'cube' }
        colormap='copper',  # 'bone', 'copper',
        line_width=10,
        scale_factor=1
    )
    mlab.axes(xlabel='x', ylabel='y', zlabel='z',ranges=(0,20,0,20,0,10),nb_labels=10)
    mlab.show()
    
def viz_3d_matplotlib(pt_3d):
    X = pt_3d[0,:]
    Y = pt_3d[1,:]
    Z = pt_3d[2,:]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X,
               Y,
               Z,
               s=1,
               cmap='gray')
    
    plt.show()
#データセットの指定
data_num=1

# pathの定義
"""current_dir=os.getcwd()
data_dir="big_data_NO_GIT/SfM_datas"
images_dir=data_dir+f"/data{data_num}/images/observatory"
calibration_file_dir = data_dir + f'/data{data_num}/calibration'"""

#images_dir=f"/home/ytpc2019a/code_ws/big_data_NO_GIT/images{data_num}"
images_dir="/home/ytpc2019a/code_ws/class_CV/final_report_SfM/webcam"
#images_dir ="big_data_NO_GIT/SfM_datas/data1/images/observatory"

# K行列 (カメラ行列3x3)を求める
# 広角カメラ
# K=np.array([[1634.30188,0,2027.12393],[0.,1639.62261,1471.23072],[0.,0.,1.]])
# WEBcam
#K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]])
# sample
#K=np.array([[3422.9, 3421.53, 3035.78], [2006.3, 0.226644, 0.128266], [0.00107347, -0.000100219, -0.00592345]])
# Kinect
K=np.array([[603.80578513,0,630.30823915],[0,603.88362465,366.13473242],[0,0,1]])
R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))
P1 = np.matmul(K, R_t_0)
P2 = np.empty((3,4))
X = np.array([])
Y = np.array([])
Z = np.array([])
# 各2枚の画像について

## 特徴量マッチングをする
images=glob.glob(images_dir+"/*.jpg")
print(images)
for i in range(len(images)-1):
    print(f"i={i}/{len(images)-1}: read img")
    img1=cv2.imread(images[i],0)
    img2=cv2.imread(images[i+1],0)
    print(f"i={i}/{len(images)-1}: ignite SIFT")
    sift = cv2.SIFT_create()  # インスタンス化
    # 特徴点の抽出
    print(f"i={i}/{len(images)-1}: detect kp")
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 以下、特徴点のマッチング
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print(f"i={i}/{len(images)-1}: choose kp")
    # 条件の良い特徴点の組みだけを残す
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    #  F行列を求める
    print(f"i={i}/{len(images)-1}: get F")
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3, 0.99)
    # src_pts = src_pts.flatten()
    # dst_pts = dst_pts.flatten()
## E=K^-1・F・Kにより、E行列を求める  <---ここの計算が理解できていない。こっち？https://tutorialmore.com/questions-1705590.htm
    try:
        E = np.matmul(np.matmul(np.transpose(K), F), K)
        print(f"i={i}/{len(images)-1}: get R,t")
        retval, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
        R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
        P2 = np.matmul(K, R_t_1)
        src_pts = np.transpose(src_pts)
        dst_pts = np.transpose(dst_pts)
        points_3d = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)
        points_3d /= points_3d[3]
        opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
        num_points = len(dst_pts[0])
        rep_error_fn(opt_variables, dst_pts, num_points)
        X = np.concatenate((X, points_3d[0]))
        Y = np.concatenate((Y, points_3d[1]))
        Z = np.concatenate((Z, points_3d[2]))

        R_t_0 = np.copy(R_t_1)
        P1 = np.copy(P2)

    except ValueError:
        print(f"i={i}/{len(images)}: Value error. Maybe match was failed. Skip the image")
        images[i+1]=images[i]

## カメラの回転R、移動t、その推定の誤差retvalを計算する。
pts_4d=[]
pts_4d.append(X)
pts_4d.append(Y)
pts_4d.append(Z)

viz_3d_matplotlib(np.array(pts_4d))