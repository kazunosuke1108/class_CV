import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize as op

def match_feature(img1,img2):
    # SIFTを用意
    sift=cv.SIFT_create()
    # 特徴点とdescriberを検出
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # マッチさせる
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []# オブジェクトの保管場所
    good2= []# オブジェクトの保管場所 drawMatchesKnnに食わせるための形式
    for m, n in matches:
        if m.distance < 0.5*n.distance:  # 厳選を実施
            good.append(m)
            good2.append([m])
    img1_pt = [list(map(int, kp1[m.queryIdx].pt)) for m in good]#マッチした１枚目の特徴点
    img2_pt = [list(map(int, kp2[m.trainIdx].pt)) for m in good]#マッチした２枚目の特徴点
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good2,
                            None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)#マッチ状況の描画
    plt.imshow(img3), plt.show()#確認のためマッチの様子を出力

    img1_pt_s=[]#重複削除後のマッチした特徴点
    img2_pt_s=[]    
    for i in range(len(img1_pt)):
        if (img1_pt[i] not in img1_pt_s) and (img2_pt[i] not in img2_pt_s):#重複の確認
            img1_pt_s.append(img1_pt[i])
            img2_pt_s.append(img2_pt[i])

    img1_pt_s=np.array(img1_pt_s)#最適化に備えnumpy形式に変換
    img2_pt_s=np.array(img2_pt_s)
    return img1_pt_s, img2_pt_s

def error_func(t):
    # 特徴点の合致における誤差関数
    answer=0
    for i in range(len(img1_pt_s)):#特徴点の組毎に足しこむ
        # 特徴点間の距離の2乗
        answer+=(np.linalg.norm(t+img2_pt_s[i]-img1_pt_s[i]))**2
    t_x.append(t[0])#プロットのための記録
    t_y.append(t[1])
    accum.append(answer)
    return answer

# Question 1 写真準備
img_a = cv.imread('exercise0509/img_data/IMG_8224.jpg', cv.IMREAD_GRAYSCALE)  
img_b = cv.imread('exercise0509/img_data/IMG_8225.jpg', cv.IMREAD_GRAYSCALE)  
img_c = cv.imread('exercise0509/img_data/IMG_8226.jpg', cv.IMREAD_GRAYSCALE)  
img_d = cv.imread('exercise0509/img_data/IMG_8227.jpg', cv.IMREAD_GRAYSCALE)  
img_e = cv.imread('exercise0509/img_data/IMG_8228.jpg', cv.IMREAD_GRAYSCALE)  
img_f = cv.imread('exercise0509/img_data/IMG_8229.jpg', cv.IMREAD_GRAYSCALE)  
img_g = cv.imread('exercise0509/img_data/IMG_8230.jpg', cv.IMREAD_GRAYSCALE)  
img_h = cv.imread('exercise0509/img_data/IMG_8231.jpg', cv.IMREAD_GRAYSCALE)  
img=[img_a,img_b,img_c,img_d,img_e,img_f,img_g,img_h]

# 結果出力のキャンバスを作成
canvas_h,canvas_w=1500,1500
canvas=np.zeros((canvas_h,canvas_w))
canvas+=255
height,width=img_a.shape

# 1枚目の写真を貼る
vector_root=np.array([300,100])
canvas[int(vector_root[0]):int(vector_root[0])+height,int(vector_root[1]):int(vector_root[1])+width]=img_a

# 写真を2枚ずつ比べ、2枚目の方だけを描画
for i in range(len(img)-1):
    # 写真の名前を定義
    img1=img[i]
    img2=img[i+1]
    # マッチする特徴点を抽出
    img1_pt_s, img2_pt_s=match_feature(img1,img2)
    # 最適化可視化のための記録リスト
    t_x=[]
    t_y=[]
    accum=[]
    #最適なベクトルtの探索
    vec=op.minimize(error_func,[0,500]).x
    #前回のベクトルに足し込むことで全体座標系での移動量を求める
    vector=np.array([vec[1],vec[0]])+vector_root
    #これから貼り付ける画像の大きさを取得
    height,width=img2.shape
    #現状の値と貼り付け画像の平均をとって貼り付ける
    canvas[int(vector[0]):int(vector[0])+height,int(vector[1]):int(vector[1])+width]=(img2+canvas[int(vector[0]):int(vector[0])+height,int(vector[1]):int(vector[1])+width])/2
    vector_root=vector
    #最適化過程のプロット
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(t_x,t_y,accum)
    plt.show()
#完成した画像を保存
canvas=cv.imwrite("exercise0509/canvas.jpg",canvas)
