import numpy as np
import cv2 as cv
import scipy.optimize as op
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def match_feature(img1, img2):  # 前回の課題と同様
    # SIFTを用意
    sift = cv.SIFT_create()
    # 特徴点とdescriberを検出
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # マッチさせる
    bf = cv.BFMatcher()
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
    # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good2,
    #                         None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # マッチ状況の描画
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
    return answer


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# 動画の読み込み
mv = cv.VideoCapture("exercise0516/prof_likes_hiroyuki3.mp4")
frame_count = int(mv.get(cv.CAP_PROP_FRAME_COUNT))  # フレーム数
size = (1080, 1920)
# size=(540,960)
size_cv = (size[1], size[0])
frame_rate = int(mv.get(cv.CAP_PROP_FPS))

# 保存形式の指定
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
save = cv.VideoWriter('exercise0516/result/stabilized.mp4',
                      fourcc, frame_rate, size_cv)

vec0_list = []
vec1_list = []

# for i in range(20):
for i in range(frame_count):
    ch, frame = mv.read()  # １フレームずつ取り出す
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  # モノクロに変換
    if i != 0:  # 初回はスキップ
        print(i)
        img1_pt_s, img2_pt_s = match_feature(frame_old, frame)  # 特徴点マッチング
        vec = op.minimize(error_func, [0, 0]).x  # ブレのベクトル最適化
        vec = [int(vec[0]), int(vec[1])]  # ベクトルの成分をintに変換
        vec0_list.append(vec[0])
        vec1_list.append(vec[1])
    frame_old = frame

# ベクトルuの平滑化処理

# LPFのパラメータ。試行錯誤的に決定。
order = 6
fs = 30.0
cutoff = 3.667
T = 5.0
n = int(T * fs)

vec0_list = butter_lowpass_filter(
    vec0_list, cutoff, fs, order)  # スタビライズベクトルの高周波成分の消去
vec1_list = butter_lowpass_filter(vec1_list, cutoff, fs, order)


mv2 = cv.VideoCapture("exercise0516/prof_likes_hiroyuki3.mp4")

# for i in range(19):
for i in range(0, frame_count-1):
    ch, frame = mv2.read()  # １フレームずつ取り出す
    if ch:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  # モノクロに変換
        vec[0] = int(vec0_list[i])
        vec[1] = int(vec1_list[i])
        y_start = 0 if vec[0] <= 0 else vec[0]  # 切り出す部分の設定
        y_end = size[0]+vec[0] if vec[0] <= 0 else size[0]
        x_start = 0 if vec[1] <= 0 else vec[1]
        x_end = size[1]+vec[1] if vec[1] <= 0 else size[1]
        tsukau_tokoro = frame[y_start:y_end, x_start:x_end]  # 画像を切り出す
        print(i, tsukau_tokoro.shape)
        canvas = np.insert(tsukau_tokoro, 0 if vec[0] <= 0 else tsukau_tokoro.shape[0], np.zeros(
            (abs(vec[0]), 1)), axis=0)  # 空白行の挿入
        canvas = np.insert(canvas, 0 if vec[1] <= 0 else tsukau_tokoro.shape[1], np.zeros(
            (abs(vec[1]), 1)), axis=1)  # 空白列の挿入
        canvas = canvas[100:canvas.shape[0]-100, 100:canvas.shape[1]-100]  # 拡大
        frame = cv.resize(canvas, size_cv)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        save.write(frame)

print("保存しました")

mv2.release()  # ファイルを閉じる
save.release()
cv.destroyAllWindows()  # ウインドウを閉じる