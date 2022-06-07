import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
# 内積の計算で、値を0で除してしまった場合の検知
import warnings
warnings.simplefilter('error')

# 点と直線の距離を求める関数


def dist_pt_line(pt, line_pt1, line_pt2):
    # 2つのベクトルの内積から角度を計算し、距離distanceを導出する
    vec_u = np.array(pt)-np.array(line_pt1)
    vec_v = np.array(line_pt2)-np.array(line_pt1)
    try:
        distance = np.linalg.norm(
            vec_u)*np.sin(np.arccos(np.inner(vec_u, vec_v)/np.linalg.norm(vec_u)/np.linalg.norm(vec_v)))
    except RuntimeWarning:  # 0で除してしまった時のための回避
        distance = 10e5
    return distance

# 図中にepilineを引く


def drawlines(img1, img2, lines, pts1, pts2):
    # img1はimg2上の点に対応するエピポーラ線を描画する画像でlinesはそれに対応するエピポーラ線
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    e = 0  # 誤差を蓄積
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color,
                        thickness=3)  # epilineを引く
        distance = dist_pt_line(pt1, (x0, y0), (x1, y1))  # 点と直線の距離を計算する
        e += distance  # 誤差を足していく
        img1 = cv2.circle(img1, tuple(pt1), int(distance), color, thickness=3)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, thickness=3)
    return img1, img2, e


# 異なる端末でも再現が確実になるように、パスの取得を強化
os.chdir("class_CV/exercise0606/images")
images_path = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("results")
results_path = os.getcwd()

# 全部で19種類の写真を試す
pic_number = 19
error_list = []
for p_num in range(1, pic_number+1):  # 各写真のセットについて
    left_path = images_path+f"/ex0606left{str(p_num)}.jpg"
    right_path = images_path+f"/ex0606right{str(p_num)}.jpg"
    error = 0
    img1 = cv2.imread(left_path, 0)
    img2 = cv2.imread(right_path, 0)
    # ポイントマッチング（以前の課題同様）
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
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    # 特徴量がきちんと検出されているかの確認は、以下を実行
    """img3 = img1
    for pt in pts1:
        img3=cv2.circle(img3,(int(pt[0]),int(pt[1])),10,(255,255,255))
    cv2.imshow('feature points in img1',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img4 = img2
    for pt in pts2:
        img4=cv2.circle(img4,(int(pt[0]),int(pt[1])),10,(255,255,255))
    cv2.imshow('feature points in img2',img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # inlier pointsだけを使用
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # 導出されたFundamental Matrixを出力
    print(f"Fundamental Matrix of picture {p_num}\n", F)

    # 右画像(二番目の画像)中の点に対応するepilineの計算をし、そのエピポーラ線を左画像に描画
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6, e1 = drawlines(img1, img2, lines1, pts1, pts2)
    # 上記と逆のことをする
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4, e2 = drawlines(img2, img1, lines2, pts2, pts1)
    # 誤差の合計を取る
    er = e1+e2
    # 特徴点の数で割った値を誤差の代表値にする
    error_list.append(er/len(pts1))
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    # epilineに加え、精度がわかりやすいように誤差もepilineの接円として表した。円が目立てば目立つほど、全体として誤差が大きいと言う感覚的理解ができるようになっている。
    plt.savefig(results_path+f"/result_pic{p_num}.png")
    plt.close()

# 誤差を写真間で比較する
plt.bar(np.arange(1, len(error_list)+1, 1), error_list, 0.8, align="center")
plt.title("error (normalized)")
plt.xlabel("picture number")
plt.ylabel("value of error")
plt.savefig(results_path+f"/error.png")
