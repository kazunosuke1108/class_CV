import cv2
for i in range (10):
    image_left=cv2.imread("exercise0613/results/left.png",0)
    image_right=cv2.imread("exercise0613/results/right.png",0)
    # image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    # image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16*i, blockSize=9)
    disparity = stereo.compute(image_left[:2300,800:],image_right[:2300,800:])
    cv2.imwrite('disparity.png',disparity)