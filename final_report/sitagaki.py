import cv2
import numpy as np


face_cascade_path = '/Users/hayashidekazuyuki/opt/anaconda3/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


#captureの準備
cap = cv2.VideoCapture(0)

def face_extract(frame):
    face_recog=False
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    for j, (x, y, w, h) in enumerate(faces):
        if w>100 and h>100:
            print(x,y)
            face=frame[y-50:y+h+50,x-100:x+w+100]
            return x,y,w,h,face
    return 0,0,0,0,np.zeros([400,400])
#起動と画面表示まで
while(1):
    #capture frameの作成
    _, frame = cap.read()
    cv2.imshow('Original', frame)
    _,_,_,_,face=face_extract(frame)
    cv2.imshow('face',face)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

# calibする
# arマーカーの向きを出力する