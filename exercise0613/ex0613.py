import numpy as np
import cv2
import matplotlib.pyplot as plt


def rectify_pair(image_left, image_right, viz=False):
    # 前回同様、特徴点抽出
    sift = cv2.SIFT_create()  # インスタンス化
    # 特徴点の抽出
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)
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
    print(image_left.shape[:2])
    # ホモグラフィを生成する
    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(
        src_pts, dst_pts, F, image_left.shape[:2])
    return F, H_left, H_right


def drawlines(img1, img2, num):
    # img1はimg2上の点に対応するエピポーラ線を描画する画像でlinesはそれに対応するエピポーラ線
    r, c = img1.shape  # 形状を把握
    # モノクロに直す
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # 以下、rectify_pairと同様。一部仕様が異なるため関数には纏めなかった（「# 違い」の部分）
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)  # 違い
    pts2 = np.int32(pts2)  # 違い
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)  # 違い
    # inlier pointsだけを使用
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    # epipolar lineの計算
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    # epipolar lineの計算
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img3 = cv2.line(img1, (x0, y0), (x1, y1), color,
                        thickness=3)  # epilineを引く
    for r, pt1, pt2 in zip(lines2, pts2, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img5 = cv2.line(img2, (x0, y0), (x1, y1), color,
                        thickness=3)  # epilineを引く
    plt.subplot(121), plt.imshow(img3)
    plt.subplot(122), plt.imshow(img5)
    # 画像の保存
    plt.savefig(
        f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/class_CV/exercise0613/results/results{num}.png")
    plt.close()
    return img3, img5


# 画像の相対パス
left_path = "exercise0606/images/ex0606left10.jpg"
right_path = "exercise0606/images/ex0606right10.jpg"


# 画像の読み込み・モノクロ変換
image_left = cv2.imread(left_path)
image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
image_right = cv2.imread(right_path)
image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
# rectification前のepipolar lineの計算
drawlines(image_left, image_right, 1)
print("result1.png renewed")
# rectificationに使うhomography等の計算（Fもここで再度計算）
F, H_left, H_right = rectify_pair(image_left, image_right)
print(image_right.shape, H_right.shape)
# homographyを用いてrectificationの結果を生成
image_left = cv2.warpPerspective(image_left, H_left, image_left.shape[:2])
image_right = cv2.warpPerspective(image_right, H_right, image_right.shape[:2])
# rectification後のepipolar lineの計算
drawlines(image_left, image_right, 2)
print("result2.png renewed")
# rectificationの結果を保存
cv2.imwrite("exercise0613/results/left.png", image_left)
cv2.imwrite("exercise0613/results/right.png", image_right)
#　保存した画像を再度読み込み（debug容易性のため）
image_left = cv2.imread("exercise0613/results/left.png", 0)
image_right = cv2.imread("exercise0613/results/right.png", 0)
# disparityの計算のためのインスタンス生成
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
# disparity画像の生成
disparity = stereo.compute(image_left[:2300, 800:], image_right[:2300, 800:])
# 結果画像の保存
cv2.imwrite('exercise0613/results/disparity.png', disparity)