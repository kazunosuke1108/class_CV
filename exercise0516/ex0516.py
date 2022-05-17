import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize as op

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

def match_feature(img1, img2):
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
    #img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good2,
    #                         None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # マッチ状況の描画
    #plt.imshow(img3), plt.show()  # 確認のためマッチの様子を出力

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

#save_all_frames('exercise0516/prof_likes_hiroyuki.mp4', 'exercise0516/frame_img', 'video_img')

frame_img_path="exercise0516/frame_img"
img=[]
fourcc = cv.VideoWriter_fourcc('m','p','4', 'v')
video  = cv.VideoWriter('exercise0516/result/ImgVideo.mp4', 0x00000020, 60.0, (2000, 1500),isColor=False)

print("start loading_img")
for filename in os.listdir(frame_img_path):
    img.append(cv.imread(frame_img_path+"/"+filename, cv.IMREAD_GRAYSCALE))

#テスト用に画像削減
img=img[:10]


vector_root = np.array([300, 300])

print("start finding vector u")
for i in range(len(img)-1):
    if i%10==0:
        print(f"img No. {i} finished")
    img1=img[i]
    img2=img[i+1]

    img1_pt_s, img2_pt_s = match_feature(img1, img2)
    vec = op.minimize(error_func, [0, 500]).x
    print(vec)

    canvas = np.zeros((2000,1500))+255
    vector = np.array([vec[1], vec[0]])+vector_root
    height, width = img2.shape
    print(height, width)
    print(canvas[int(vector[0]):int(vector[0])+height, int(vector[1]):int(vector[1])+width].shape)
    canvas[int(vector[0]):int(vector[0])+height, int(vector[1]):int(vector[1])+width] =img2
    stabilized_img=cv.imwrite("exercise0516/result/temp_img.jpg", canvas)
    video.write(stabilized_img)

video.release()
