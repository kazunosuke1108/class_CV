import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


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
    # plt.imshow(img3), plt.show()  # 確認のためマッチの様子を出力

    img1_pt_s = []  # 重複削除後のマッチした特徴点
    img2_pt_s = []
    for i in range(len(img1_pt)):
        if (img1_pt[i] not in img1_pt_s) and (img2_pt[i] not in img2_pt_s):  # 重複の確認
            img1_pt_s.append(img1_pt[i])
            img2_pt_s.append(img2_pt[i])

    img1_pt_s = np.array(img1_pt_s)  # 最適化に備えnumpy形式に変換
    img2_pt_s = np.array(img2_pt_s)
    return img1_pt_s, img2_pt_s


def error_func(t):
    # 特徴点の合致における誤差関数
    answer = 0
    for i in range(len(img1_pt_s)):  # 特徴点の組毎に足しこむ
        # 特徴点間の距離の2乗
        answer += (np.linalg.norm(t+img2_pt_s[i]-img1_pt_s[i]))**2
    t_x.append(t[0])  # プロットのための記録
    t_y.append(t[1])
    accum.append(answer)
    return answer

save_dir='class_CV/final_report/kabe_images'
files=sorted(glob.glob(save_dir+"/*"))

pictures=[]
for file in files:
    print(file)
    pictures.append(cv2.imread(file,0))

canvas_h, canvas_w = 2000, 5000
canvas = np.zeros((canvas_h, canvas_w))
canvas += 255
height, width = pictures[0].shape

# 1枚目の写真を貼る
vector_root = np.array([300, 100])
canvas[int(vector_root[0]):int(vector_root[0])+height,
       int(vector_root[1]):int(vector_root[1])+width] = pictures[0]

# 写真を2枚ずつ比べ、2枚目の方だけを描画
for i in range(len(pictures)-1):
    # 写真の名前を定義
    img1 = pictures[i]
    img2 = pictures[i+1]
    # マッチする特徴点を抽出
    img1_pt_s, img2_pt_s = match_feature(img1, img2)
    # 最適化可視化のための記録リスト
    t_x = []
    t_y = []
    accum = []
    # 最適なベクトルtの探索
    vec = op.minimize(error_func, [0, 500]).x
    # 前回のベクトルに足し込むことで全体座標系での移動量を求める
    vector = np.array([vec[1], vec[0]])+vector_root
    # これから貼り付ける画像の大きさを取得
    height, width = img2.shape
    print(int(vector[1]),int(vector[1])+width)
    print(img2.shape,canvas[int(vector[0]):int(vector[0])+height, int(vector[1]):int(vector[1])+width].shape)
    # 現状の値と貼り付け画像の平均をとって貼り付ける
    try:
        canvas[int(vector[0]):int(vector[0])+height, int(vector[1]):int(vector[1])+width] = (img2 +
                                                                                         canvas[int(vector[0]):int(vector[0])+height, int(vector[1]):int(vector[1])+width])/2
        vector_root = vector
        # 最適化過程のプロット
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(t_x, t_y, accum)
        # plt.show()
    except ValueError:
        pictures[i+1]=pictures[i]
# 完成した画像を保存
    cv2.imwrite("class_CV/final_report/canvas.jpg", canvas)
