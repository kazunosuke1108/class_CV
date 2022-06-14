import numpy as np
import cv2
import matplotlib.pyplot as plt

def rectify_pair(image_left, image_right, viz=False):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    # find the fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3, 0.99)
    src_pts = src_pts.flatten()
    dst_pts = dst_pts.flatten()
    print(image_left.shape[:2])

    # rectify the images, produce the homographies: H_left and H_right
    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(
        src_pts, dst_pts, F, image_left.shape[:2])

    return F, H_left, H_right


def drawlines(img1, img2,num):
    # img1はimg2上の点に対応するエピポーラ線を描画する画像でlinesはそれに対応するエピポーラ線
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
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
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # inlier pointsだけを使用
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
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
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.savefig(f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/class_CV/exercise0613/results/test{num}.png")
    plt.close()
    return img3, img5

left_path="exercise0606/images/ex0606left10.jpg"
right_path="exercise0606/images/ex0606right10.jpg"



image_left=cv2.imread(left_path)
image_left=cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
image_right=cv2.imread(right_path)
image_right=cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
drawlines(image_left,image_right,1)
print("result1.png renewed")

F,H_left,H_right=rectify_pair(image_left,image_right)
print(image_right.shape,H_right.shape)
image_left=cv2.warpPerspective(image_left,H_left,image_left.shape[:2])
image_right=cv2.warpPerspective(image_right,H_right,image_right.shape[:2])
drawlines(image_left,image_right,2)
print("result2.png renewed")

print(H_left,H_right)
cv2.imwrite("exercise0613/results/left.png",image_left)
cv2.imwrite("exercise0613/results/right.png",image_right)
# image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
# image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(image_left[:2300,800:],image_right[:2300,800:])
plt.imshow(disparity,'gray')
plt.show()