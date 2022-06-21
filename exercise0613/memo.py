import cv2
for i in range (10):
    # image_left=cv2.imread("class_CV/exercise0613/results/left.png",0)
    # image_right=cv2.imread("class_CV/exercise0613/results/right.png",0)
    image_left=cv2.imread("/home/ytpc2019a/code_ws/class_CV/exercise0613/images/Tsukuba_L.png")
    image_right=cv2.imread("/home/ytpc2019a/code_ws/class_CV/exercise0613/images/Tsukuba_R.png")
    image_left=cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    image_right=cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
    print(type(image_left))
    disparity = stereo.compute(image_left,image_right)
    cv2.imwrite('/home/ytpc2019a/code_ws/disparity.png',disparity)