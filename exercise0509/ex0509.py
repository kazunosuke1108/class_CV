import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pprint
import scipy.optimize as op

def match_feature(img1,img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print("matches:",matches)
    img1_pt = [list(map(int, kp1[m.queryIdx].pt)) for m in matches[0]]
    img2_pt = [list(map(int, kp2[m.trainIdx].pt)) for m in matches[0]]
    print(img1_pt)
    print(img2_pt)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:  # ここを書き換えれば良いと思われる
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good,
                            None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    cv.imwrite('result.jpg', img3)
    
    return [img1_pt,img2_pt,good]

def extract_duplicate(feature_set):
    answer=[]
    answer_distance=[]
    for feature in feature_set:
        if feature.distance not in answer_distance:
            answer.append(feature)
            answer_distance.append(feature.distance)
    return answer


# Question 1
img1 = cv.imread('exercise0509/img_data/IMG_8208.jpg', cv.IMREAD_GRAYSCALE)  # import queryImage
img2 = cv.imread('exercise0509/img_data/IMG_8209.jpg', cv.IMREAD_GRAYSCALE)  # import trainImage
img3 = cv.imread('exercise0509/img_data/IMG_8210.jpg', cv.IMREAD_GRAYSCALE)  # import queryImage
img4 = cv.imread('exercise0509/img_data/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)  # import trainImage

img=[img1,img2,img3,img4]

# Question 2
kp_list=[]
feature_set_list=[]
for i in range(0,3):
    matching=match_feature(img[i],img[i+1])
    feature_set_list.append(matching[2])
    kp_list.append(matching[0])

# Question 3
for feature_set in feature_set_list:
    flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]
    feature_set=flatten(feature_set)
    feature_set=extract_duplicate(feature_set)

# Question 4
"""
t_j : t枚目の画像の座標系の位置
x_ij: t枚目の画像の中にある、i番目の特徴点の位置
x_i : 全体の座標系で見た時の、x_ijの位置
"""
print(kp_list)

def evaluate_func0(t):
    global new_kp_list
    kps_1=new_kp_list[0]
    kps_2=new_kp_list[1]
    t_x=t[0]
    t_y=t[1]
    func=0
    for i in range(len(kps_2)):
        func+=np.linalg.norm([t_x+kps_2[i][0]-kps_1[i][0],t_y+kps_2[i][1]-kps_1[i][1]])
    return func

def evaluate_func1(t):
    global kp_list
    kp_list=flatten(kp_list)
    kps_1=kp_list[1]
    kps_2=kp_list[2]
    t_x=t[0]
    t_y=t[1]
    func=0
    for i in range(len(kps_2)):
        func+=np.linalg.norm([t_x+kps_2[i][0]-kps_1[i][0],t_y+kps_2[i][1]-kps_1[i][1]])
    return func

def evaluate_func2(t):
    global new_kp_list
    kps_1=kp_list[2]
    kps_2=new_kp_list[3]
    t_x=t[0]
    t_y=t[1]
    func=0
    for i in range(len(kps_2)):
        func+=np.linalg.norm([t_x+kps_2[i][0]-kps_1[i][0],t_y+kps_2[i][1]-kps_1[i][1]])
    return func


vector0=op.minimize(evaluate_func0,(500,0)).x+np.array([750,1250])
vector1=op.minimize(evaluate_func1,(500,0)).x+vector0
vector2=op.minimize(evaluate_func2,(500,0)).x+vector1

print(vector0,vector1,vector2)

#ブランク画像
height = 1500
width = 2500
blank = np.zeros((height, width, 3))
kansei=cv.imwrite("kansei.png",blank)
kansei=cv.imread("kansei.png", cv.IMREAD_GRAYSCALE)
height,width=img1.shape[:2]
print(kansei.shape)
kansei[0:height,0:width]=img1
print(kansei.shape)
print(kansei[int(vector0[0]):int(vector0[0])+height,int(vector0[1]):int(vector0[1])+width].shape)
kansei[int(vector0[0]):int(vector0[0])+height,int(vector0[1]):int(vector0[1])+width]=img2
cv.imshow("kansei.png",kansei)
kansei=cv.imwrite("kansei.png",kansei)
