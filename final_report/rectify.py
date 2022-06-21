# -*- coding: utf-8 -*-
import cv2
import cv2.aruco as aruco
import numpy as np
from math import pi
arucoMarkerLength = 0.05

class AR():

    def __init__(self, videoPort, cameraMatrix, distortionCoefficients):
        self.cap = cv2.VideoCapture(videoPort)
        #self.cameraMatrix = np.load(cameraMatrix)
        #self.distortionCoefficients = np.load(distortionCoefficients)
        self.cameraMatrix = cameraMatrix
        self.distortionCoefficients = distortionCoefficients
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    def find_ARMarker(self):
        self.ret, self.frame = self.cap.read()
        if len(self.frame.shape) == 3:
            self.Height, self.Width, self.channels = self.frame.shape[:3]
        else:
            self.Height, self.Width = self.frame.shape[:2]
            self.channels = 1
        self.halfHeight = int(self.Height / 2)
        self.halfWidth = int(self.Width / 2)
        self.corners, self.ids, self.rejectedImgPoints = aruco.detectMarkers(self.frame, self.dictionary)
        #corners[id0,1,2...][][corner0,1,2,3][x,y]
        aruco.drawDetectedMarkers(self.frame, self.corners, self.ids, (0,255,0))

    def show(self):
        cv2.imshow("result", self.frame)

    def get_exist_Marker(self):
        return len(self.corners)

    def is_exist_marker(self, i):
        num = self.get_exist_Marker()
        if i >= num:
            return False
        else:
            return True

    def release(self):
        self.cap.release()

    # マーカー頂点の座標を取得
    def get_ARMarker_points(self, i):
        if self.is_exist_marker(i):
            return self.corners[i]

    def get_average_point_marker(self, i):
        if self.is_exist_marker(i):
            points = self.get_ARMarker_points(i)
            points_reshape = np.reshape(np.array(points), (4, -1))
            G = np.mean(points_reshape, axis = 0)
            cv2.circle(self.frame, (int(G[0]), int(G[1])), 10, (255, 255, 255), 5)
            return G[0], G[1]

    def get_ARMarker_pose(self, i):
        if self.is_exist_marker(i):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(self.corners[i], arucoMarkerLength, self.cameraMatrix, self.distortionCoefficients)
            self.frame = aruco.drawAxis(self.frame, self.cameraMatrix, self.distortionCoefficients, rvec, tvec, 0.1)
            return rvec, tvec

    def get_degrees(self, i):
        if self.is_exist_marker(i):
            rvec, tvec, = self.get_ARMarker_pose(i)
            (roll_angle, pitch_angle, yaw_angle) =  rvec[0][0][0]*180/pi, rvec[0][0][1]*180/pi, rvec[0][0][2]*180/pi
            if pitch_angle < 0:
                roll_angle, pitch_angle, yaw_angle = -roll_angle, -pitch_angle, -yaw_angle
            return roll_angle, pitch_angle, yaw_angle
            
if __name__ == '__main__':

    camera_matrix = np.matrix([[538.0, 0.0, 306.0], [0.0, 538.0, 212.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.02524192, -0.15412883, -0.00789069, -0.00260461,  0.04049524])
    myCap = AR(0, camera_matrix, distortion)
    while True:
        myCap.find_ARMarker()
        myCap.get_average_point_marker(0)
        print(myCap.get_degrees(0))
        myCap.show()
        if cv2.waitKey(1) > 0:
            myCap.release()
            cv2.destroyAllWindows()
            break