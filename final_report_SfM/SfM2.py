# 

from pprint import pprint
from tracemalloc import start
from typing import Type
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
from camera import camera
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.widgets as wg

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
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good2,
    #                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # マッチ状況の描画
    # img3=cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
    # plt.imshow(img3), plt.show()  # 確認のためマッチの様子を出力

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

def crt_checker(img1,img2,img3,img4,img5,img6):
    global approval
    approval=False
    fig=plt.figure(figsize=(10,10))
    imgs=[img1,img2,img3,img4,img5,img6]
    titles=["input 1","input 2","rectified 1","rectified 2","disparity raw","disparity"]
    X=4
    Y=2
    for i,(img,title) in enumerate(zip(imgs,titles)):
        ax=fig.add_subplot(X,Y,i+1)
        ax.set_title(title)
        plt.imshow(img)
    del ax
    axcolor = 'lightgoldenrodyellow'
    ax1 = plt.axes([0.2, 0.1, 0.15, 0.05])
    ax2 = plt.axes([0.7, 0.1, 0.15, 0.05])
    btn1 = wg.Button(ax1, 'approve', color=axcolor, hovercolor='0')
    btn2 = wg.Button(ax2, 'deny', color=axcolor, hovercolor='1')
    def approve(self):
        print("matching succeeded")
        plt.close('all')
        global approval
        approval=True
    def deny(self):
        print("matching failed")
        plt.close('all')
        global approval
        approval=False


    approval=btn1.on_clicked(approve)
    approval=btn2.on_clicked(deny)
    plt.show()
    return approval


def interpolate(i, imgL, imgR, disparity_raw):
    img_interp = np.zeros_like(imgL)  # 描画のキャンバスを用意
    for y in range(imgL.shape[0]):  # 画像の縦方向
        for x1 in range(imgL.shape[1]):  # 画像の横方向
            # if y % 100 == 0 and x1 % 100 == 0:
            #     print("current process: ", y, x1)
            x2 = int(x1 - disparity_raw[y, x1])  # 左側画像に対応する点の右側座標を取得
            x_i = int((2 - i) * x1 + (i - 1) * x2)  # 左側画像と右側座標の間の位置を決定
            # 移動先が画像のサイズを超えていないか確認
            if 0 <= x_i < img_interp.shape[1] and 0 <= x2 < imgR.shape[1]:
                img_interp[y, x_i,:] = (2 - i) * imgL[y, x1,:] + (i - 1) * imgR[y, x2,:]  # 　移動先の画素値を決定
    return img_interp.astype(np.uint8)

def match_to_disparity(img1,img2):
    img1_pt_s, img2_pt_s, F = match_feature(img1,img2)
    E = K.T.dot(F).dot(K)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(img1_pt_s, img2_pt_s, F, img1.shape[:2])
    img1r = cv2.warpPerspective(img1, H1, [img1.shape[1], img1.shape[0]])
    img2r = cv2.warpPerspective(img2, H2, [img2.shape[1], img2.shape[0]])
    cv2.imwrite(current_dir+"/results/img1r.jpg", img1r)
    cv2.imwrite(current_dir+"/results/img2r.jpg", img2r)
    disparity_raw,disparity=crt_disparity(img1r,img2r)
    return img1,img2,img1r,img2r,disparity_raw,disparity 
# 
os.chdir("final_report_SfM")
current_dir = os.getcwd()
print(current_dir) # /home/ytpc2019a/code_ws/class_CV/final_report_SfM

images_path=sorted(glob.glob(current_dir+"/images/images3/*"))

good_images=sorted(glob.glob(current_dir+"/images/good_images/*"))
print(good_images)

last_good_image=os.path.basename(good_images[-1])
start_no=0
for i,path in enumerate(images_path):
    if last_good_image in path:
        start_no=i
        break

images_path=images_path[start_no+1:]
print("start with good image : ",last_good_image)
# parameter
K=np.array([[842.50011162,0.,578.89029916],[0.,801.01078582,246.00138272],[0.,0.,1.]])


"""
# 最初の1セット
j=0
for i in np.arange(1,len(images_path)):
    j+=1
    print("comparing :",os.path.basename(sorted(glob.glob(current_dir+"/images/good_images/*"))[-1]),os.path.basename(images_path[i]))
    img1=cv2.imread(sorted(glob.glob(current_dir+"/images/good_images/*"))[-1])
    img2=cv2.imread(images_path[i])
    img2_name=os.path.basename(images_path[i])
    try:
        img1,img2,img1r,img2r,disparity_raw,disparity=match_to_disparity(img1,img2)
        if int(np.average(img1r[0][640]))==0 or int(np.average(img2r[0][640]))==0:
            check=False
        else:
            try:
                check=crt_checker(img1,img2,img1r,img2r,disparity_raw,disparity)
            except cv2.error:
                print("### CV2 ERROR ###")
                check=False
    except TypeError:
        check=False
        print("### TYPE ERROR ###")
    except ValueError:
        check=False
        print('### ValueError ###')
    except cv2.error:
        check=False
        print("### CV2 ERROR ###")
    print(check)

    if check or j>10:
        j=0
        good_images.append(images_path[i])
        cv2.imwrite(current_dir+f"/images/good_images/{img2_name}",img2)
"""
print(images_path)
print(good_images)


interpolated_images=sorted(glob.glob(current_dir+"/results/view_interpolation/*"))

for i in range(len(good_images)-1):
    img1=cv2.imread(good_images[i])
    img2=cv2.imread(good_images[i+1])
    # view interpolationを実行
    img1_no=int(os.path.basename(good_images[i])[:-4])
    img2_no=int(os.path.basename(good_images[i+1])[:-4])

    no_calc=False
    for existing in interpolated_images:
        if str(img1_no) in existing:
            no_calc=True
            print("already calculated : ",img1_no)
            break

    if no_calc:
        continue            

    diff=int((img2_no-img1_no)/10)
    interpolate_list=[1.0,]
    print(diff)
    gap=1/diff
    while interpolate_list[-1]+gap<1.991:
        interpolate_list.append(interpolate_list[-1]+gap)
    print(interpolate_list)
    try:
        img1,img2,img1r,img2r,disparity_raw,disparity=match_to_disparity(img1,img2)
        for ratio in interpolate_list:
            print(f"interpolation : img={img1_no} ratio={ratio}")
            img_interp = interpolate(ratio,img1r, img2r, disparity_raw)
            cv2.imwrite(current_dir+f"/results/view_interpolation/{img1_no}_{str(ratio)[0]}-{str(ratio)[2:4].ljust(2,'0')}.jpg", img_interp)
    except TypeError:
        print("### TYPE ERROR ###")
    except cv2.error:
        print("### CV2 ERROR ###")

"""
for i in range(len(images_path)-1):
    img1=cv2.imread(images_path[i])
    img2=cv2.imread(images_path[i+1])
    img1_pt_s, img2_pt_s, F = match_feature(img1,img2)
    E = K.T.dot(F).dot(K)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(img1_pt_s, img2_pt_s, F, img1.shape[:2])
    img1r = cv2.warpPerspective(img1, H1, [img1.shape[1], img1.shape[0]])
    img2r = cv2.warpPerspective(img2, H2, [img2.shape[1], img2.shape[0]])
    cv2.imwrite(current_dir+"/results/img1r.jpg", img1r)
    cv2.imwrite(current_dir+"/results/img2r.jpg", img2r)
    disparity_raw,disparity=crt_disparity(img1r,img2r)
    check=crt_checker(img1,img2,img1r,img2r,disparity_raw,disparity)
    print(check)
"""

"""
アイデアメモ
・できてる・できていないを表示させる
→できていたら、その２枚目を次の１枚目にして、続ける。よかったやつはよかったやつとして残す。
→できていなかったら、２枚目を次の画像に変える。
・できているものだけでSfMをやるか、view morphingをするかして、最終的に14棟をぐるっと一周

並んだ②枚を出す
img1が前で、img2が後ろ
・マッチできていたら
    img2を登録する
    img1は登録済みのはず
    img2をimg1にする
・できてなかったら
    img2は飛ばす
    img1は保持したまま、img2を次の画像にする



view interpolationでgood imagesに入っていない画像を復元する

00300.pngと00340.pngの間には,10,20,30の3枚があるはず。
(340-300)/10-1=3
1.0,1.25,1.5,1.75,2.0
のはず。
1/(1+3)=0.25

"""

