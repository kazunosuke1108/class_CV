# 

from pprint import pprint
import numpy as np
import glob
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
        if m.distance < 0.3*n.distance:  # 厳選を実施
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
    F, mask = cv2.findFundamentalMat(img1_pt_s, img2_pt_s, cv2.RANSAC, 3, 0.99)
    return img1_pt_s, img2_pt_s, F

def crt_disparity(img1r,img2r,):
    stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=9)
    disparity_raw = stereo.compute(img1r, img2r)
    cv2.imwrite(current_dir+"/results/img_disp_raw.jpg", disparity_raw)
    disparity = (disparity_raw-disparity_raw.min()) / (disparity_raw.max()-disparity_raw.min())*256  # 正規化（0から255へ）
    kernel = np.ones((10, 10), np.float32)/100
    disparity = cv2.filter2D(disparity, -1, kernel)
    cv2.GaussianBlur(disparity, (299, 299), 0)
    kernel = np.ones((10, 10), np.float32)/100
    disparity = cv2.filter2D(disparity, -1, kernel)
    cv2.GaussianBlur(disparity, (299, 299), 0)
    cv2.imwrite(current_dir+"/results/img_disp.jpg", disparity)
    
    return disparity_raw,disparity

# 
os.chdir("final_report_SfM")
current_dir = os.getcwd()
print(current_dir) # /home/ytpc2019a/code_ws/class_CV/final_report_SfM

final_path=sorted(glob.glob(current_dir+"/images/*"))

print(final_path)

# parameter
K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]])



for i in range(len(final_path)-1):
    img1=cv2.imread(final_path[i])
    img2=cv2.imread(final_path[i+1])
    img1_pt_s, img2_pt_s, F = match_feature(img1,img2)
    E = K.T.dot(F).dot(K)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(img1_pt_s, img2_pt_s, F, img1.shape[:2])
    img1r = cv2.warpPerspective(img1, H1, [img1.shape[1], img1.shape[0]])
    img2r = cv2.warpPerspective(img2, H2, [img2.shape[1], img2.shape[0]])
    cv2.imwrite(current_dir+"/results/img1r.jpg", img1r)
    cv2.imwrite(current_dir+"/results/img2r.jpg", img2r)
    disparity_raw,disparity=crt_disparity(img1r,img2r)
"""
アイデアメモ
・できてる・できていないを表示させる
→できていたら、その２枚目を次の１枚目にして、続ける。よかったやつはよかったやつとして残す。
→できていなかったら、２枚目を次の画像に変える。
"""