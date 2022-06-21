import cv2
import numpy as np



face_cascade_path = 'class_CV/final_report/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
sift = cv2.SIFT_create()  # インスタンス化


#captureの準備
cap1 = cv2.VideoCapture(0)
print(cap1.isOpened())
cap2 = cv2.VideoCapture(2)
print(cap2.isOpened())


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)

def force_show(name,frame):
    try:
        cv2.imshow(name,frame)
    except cv2.error as e:
        print(f"{name} imshow failed: ",e)

def face_extract(frame):
    face_recog=False
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    for j, (x, y, w, h) in enumerate(faces):
        if w>100 and h>100:
            print("face_size: ",x,y)
            face=frame[y-50:y+h+50,x-100:x+w+100]
            return x,y,w,h,face
    print("face detection failed")
    return 0,0,0,0,np.zeros([400,400])

def gen_disparity(frame1,frame2):
    # 特徴点の抽出~
    print("-----------------------------------here-----------------------------------")
    print(frame1.shape)
    frame2=cv2.resize(frame2,(frame1.shape[1],frame1.shape[0]))
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)
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
    print("num of feature",dst_pts.shape)
    try:
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3, 0.99)
        src_pts = src_pts.flatten()
        dst_pts = dst_pts.flatten()
        # ホモグラフィを生成する
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(
            src_pts, dst_pts, F, frame1.shape[:2])
        frame1_rect = cv2.warpPerspective(frame1, H1, frame1.shape[:2])
        frame2_rect = cv2.warpPerspective(frame2, H2, frame2.shape[:2])
        force_show('frame1_rect',frame1_rect)
        force_show('frame2_rect',frame2_rect)
        disparity = stereo.compute(frame1_rect, frame2_rect)      
        print(disparity.shape)  
        cv2.imshow('disparity',disparity)
        return disparity
    except cv2.error:
        print("failed to create disparity")
        return np.zeros([400,400])
        

#起動と画面表示まで
while(1):
    #cap1ture frame1の作成
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        frame2=cv2.resize(frame2,(frame1.shape[1],frame1.shape[0]))
        # force_show('frame1',frame1)
        # force_show('frame2',frame2)
        _,_,_,_,face1=face_extract(frame1)
        _,_,_,_,face2=face_extract(frame2)
        force_show('face1',face1)
        force_show('face2',face2)
        try:
            disparity=gen_disparity(face1,face2)
            force_show('disparity',disparity)
        except cv2.error:
            pass
        # except TypeError as e:
        #     print("disparity failed",e)
    else:
        print("failed to imread")
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap1.release()
cap2.release()
# calibする
# arマーカーの向きを出力する