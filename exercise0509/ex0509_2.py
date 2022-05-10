import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize as op

def match_feature(img1,img2):
    sift=cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    good2= []
    for m, n in matches:
        if m.distance < 0.5*n.distance:  # ここを書き換えれば良いと思われる
            good.append(m)
            good2.append([m])
    img1_pt = [list(map(int, kp1[m.queryIdx].pt)) for m in good]
    img2_pt = [list(map(int, kp2[m.trainIdx].pt)) for m in good]
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good2,
                            None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    
    return img1_pt, img2_pt


t_x=[]
t_y=[]
accum=[]
def error_func(t):
    answer=0
    for i in range(len(img1_pt_s)):
        answer+=np.linalg.norm(t+img2_pt_s[i]-img1_pt_s[i])
    t_x.append(t[0])
    t_y.append(t[1])
    accum.append(answer)
    return answer

# Question 1
img1 = cv.imread('exercise0509/img_data/IMG_8208.jpg', cv.IMREAD_GRAYSCALE)  # import Image
img2 = cv.imread('exercise0509/img_data/IMG_8209.jpg', cv.IMREAD_GRAYSCALE)  # import Image
img3 = cv.imread('exercise0509/img_data/IMG_8210.jpg', cv.IMREAD_GRAYSCALE)  # import Image
img4 = cv.imread('exercise0509/img_data/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)  # import Image

img=[img1,img2,img3,img4]

# Question 2

img1_pt, img2_pt=match_feature(img[0],img[1])

# Question 3
img1_pt_s=[]
img2_pt_s=[]    
for i in range(len(img1_pt)):
    if (img1_pt[i] not in img1_pt_s) and (img2_pt[i] not in img2_pt_s):
        img1_pt_s.append(img1_pt[i])
        img2_pt_s.append(img2_pt[i])

img1_pt_s=np.array(img1_pt_s)
img2_pt_s=np.array(img2_pt_s)


# Question 4
# Question 6,7
vector00=np.array([300,500])
vector01=np.array([op.minimize(error_func,[0,500]).x[1],op.minimize(error_func,[0,500]).x[0]])+vector00

canvas_h=1500
canvas_w=2500
canvas=np.zeros((canvas_h,canvas_w))
img_h,img_w=img1.shape[:2]
canvas[int(vector00[0]):int(vector00[0])+img_h,int(vector00[1]):int(vector00[1])+img_w]=img1
canvas[int(vector01[0]):int(vector01[0])+img_h,int(vector01[1]):int(vector01[1])+img_w]=img2+canvas[int(vector01[0]):int(vector01[0])+img_h,int(vector01[1]):int(vector01[1])+img_w]/2
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(t_x,t_y,accum)
plt.show()

canvas=cv.imwrite("canvas.jpg",canvas)
# Check the process of optimization

"""
Ex 8.2: Panography.

Create the kind of panograph discussed in Section 8.1.2 and commonly found on the web.

(you can freely modify the following guideline.)

2. Use the feature detector, descriptor, and matcher developed in the exercise on May 2 (or existing software) to match features among the images.

3. Turn each connected component of matching features into a track, i.e., assign a unique index i to each track, discarding any tracks that are inconsistent (contain two different features in the same image).

4. Compute a global translation for each image using Equation (8.12) explained in 8.1.2 of the reference book.

5. (Optional) Since your matches probably contain errors, turn the above least-square metric into a robust metric (8.25) and re-solve your system using iteratively reweighted least squares.

6. Compute the size of the resulting composite canvas and resample each image into its final position on the canvas. (Keeping track of bounding boxes will make this more efficient.)

7. Average all of the images, or choose some kind of ordering and implement translucent over compositing (3.8).
"""