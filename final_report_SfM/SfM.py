import os
import cv2
import numpy as np
from scipy.optimize import least_squares

#データセットの指定
data_num=0

# pathの定義
current_dir=os.getcwd()
images_dir=current_dir+f"/data{data_num}/images"


# K行列 (カメラ行列3x3)を求める

# 各2枚の画像について
## 特徴量マッチングをする
## F行列を求める
## E=K^-1・F・Kにより、E行列を求める  <---ここの計算が理解できていない。こっち？https://tutorialmore.com/questions-1705590.htm
## カメラの回転R、移動t、その推定の誤差retvalを計算する。
