import cv2
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import time
class wallpaper():
    def __init__(self,cam_num):
        ignition=time.time()
        cap = cv2.VideoCapture(cam_num)
        
        if not cap.isOpened():
            print(f"camera no.{cam_num} not found")
            return 
        
        # dammy sequence
        while time.time()-ignition<3:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("current perspective", frame)

        i=0

        ret, frame = cap.read()
        if ret:    
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            key = cv2.waitKey(1) & 0xFF
            canvas,frame_old=self.draw_init(frame)
        while True:
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                cv2.imshow("current perspective", frame)
                key = cv2.waitKey(1) & 0xFF
                if i%10==0:
                    canvas,frame_old=self.draw(frame_old,frame,canvas)
                    cv2.imwrite("class_CV/final_report/canvas.jpg", canvas)
                if key == ord('c'):
                    canvas,frame_old=self.draw_init(frame)
                    cv2.imwrite("class_CV/final_report/canvas.jpg", canvas)

                elif key == ord('q'):
                    break
        i+=1


    def draw_init(self,frame):
        canvas_h, canvas_w = 2000, 5000
        canvas = np.zeros((canvas_h, canvas_w))
        canvas += 255
        height, width = frame.shape
        self.vector_root = np.array([300, 100])
        canvas[int(self.vector_root[0]):int(self.vector_root[0])+height,
            int(self.vector_root[1]):int(self.vector_root[1])+width] =frame
        cv2.imwrite("class_CV/final_report/canvas.jpg", canvas)
        print("here")
        return canvas,frame

    def draw(self,frame_old,frame,canvas):
        self.frame_old_pt_s, self.frame_pt_s = self.match_feature(frame_old, frame)
        vec = op.minimize(self.error_func, [0, 500]).x
        self.vector = np.array([vec[1], vec[0]])+self.vector_root
        height, width = frame.shape
        try:
            canvas[int(self.vector[0]):int(self.vector[0])+height, int(self.vector[1]):int(self.vector[1])+width] = (frame +
                                                                                            canvas[int(self.vector[0]):int(self.vector[0])+height, int(self.vector[1]):int(self.vector[1])+width])/2
            self.vector_root = self.vector
            frame_old=frame
        except:
            pass
        return canvas,frame_old

    def match_feature(self, frame_old, frame):
        # SIFT?????????
        sift = cv2.SIFT_create()
        # ????????????describer?????????
        kp1, des1 = sift.detectAndCompute(frame_old, None)
        kp2, des2 = sift.detectAndCompute(frame, None)
        # ??????????????????
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []  # ?????????????????????????????????
        good2 = []  # ????????????????????????????????? drawMatchesKnn??????????????????????????????
        for m, n in matches:
            if m.distance < 0.5*n.distance:  # ???????????????
                good.append(m)
                good2.append([m])
        frame_old_pt = [list(map(int, kp1[m.queryIdx].pt))
                for m in good]  # ????????????????????????????????????
        frame_pt = [list(map(int, kp2[m.trainIdx].pt))
                for m in good]  # ????????????????????????????????????
        img3 = cv2.drawMatchesKnn(frame_old, kp1, frame, kp2, good2,
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # ????????????????????????
        # plt.imshow(img3), plt.show()  # ??????????????????????????????????????????

        self.frame_old_pt_s = []  # ??????????????????????????????????????????
        frame_pt_s = []
        for i in range(len(frame_old_pt)):
            if (frame_old_pt[i] not in self.frame_old_pt_s) and (frame_pt[i] not in frame_pt_s):  # ???????????????
                self.frame_old_pt_s.append(frame_old_pt[i])
                frame_pt_s.append(frame_pt[i])

        self.frame_old_pt_s = np.array(self.frame_old_pt_s)  # ??????????????????numpy???????????????
        frame_pt_s = np.array(frame_pt_s)
        return self.frame_old_pt_s, frame_pt_s

    def error_func(self,t):
        # ??????????????????????????????????????????
        answer = 0
        for i in range(len(self.frame_old_pt_s)):  # ?????????????????????????????????
            # ????????????????????????2???
            answer += (np.linalg.norm(t+self.frame_pt_s[i]-self.frame_old_pt_s[i]))**2
        return answer
    def cut_blank(self):
        array=cv2.imread('/home/ytpc2019a/code_ws/class_CV/final_report/canvas.jpg')

        pass


wallpaper(0)

