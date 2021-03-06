from turtle import st
import cv2
import time
import os
from datetime import datetime
import shutil


def camera(save_dir_path,cam_num):
    img_dir=save_dir_path
    #img_dir="class_CV/final_report_SfM/img_calib"
    try:
        shutil.rmtree(img_dir)
    except FileNotFoundError:
        pass
    os.mkdir(img_dir)

    ignition=time.time()
    cap1 = cv2.VideoCapture(cam_num)
    print("camera recognized:",cap1.isOpened())
    print("Camera ignitting...")
    while time.time()-ignition<3:
        ret, frame = cap1.read()
        if ret:
            cv2.imshow("current perspective", frame)



    print("camera ready")


    i=1
    while(1):
        ret1, frame1 = cap1.read()
        if ret1 :
            cv2.imshow("frame1",frame1)
        k = cv2.waitKey(1) & 0xFF
        if k==ord("q"):
            cv2.destroyAllWindows()
            break
        if k==ord("c"):
            print("save_img")
            now=str(datetime.now().day)+str(datetime.now().hour)+"_"+str(datetime.now().minute)+"_"+str(datetime.now().second)
            cv2.imwrite(img_dir+"/img"+str(i)+".jpg",frame1)
            i+=1
    cap1.release()
    # cap2.release()