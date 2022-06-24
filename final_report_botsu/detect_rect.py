import cv2
import numpy as np

# ファイルを読み込み グレースケール化
img = cv2.imread('/Users/hayashidekazuyuki/Desktop/Git_Win_Air/single-view-metrology-python/img/onepiece.jpg', cv2.IMREAD_GRAYSCALE)

# しきい値指定によるフィルタリング
_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

# 輪郭を抽出
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_DUPLEX

# 図形の数の変数
triangle = 0
rectangle = 0
pentagon = 0
oval = 0
circle = 0

# 図形の設定
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 3:
        triangle +=1
        cv2.putText(img, "triangle{}".format(triangle),  (x, y), font, 0.8, (0))
        
    elif len(approx) == 4:
        rectangle +=1
        cv2.putText(img, "rectangle{}".format(rectangle),  (x, y), font, 0.8, (0))
                
    elif len(approx) == 5:
        pentagon +=1
        cv2.putText(img, "pentagon{}".format(pentagon),  (x, y), font, 0.8, (0))
        
    elif 6 < len(approx) < 14:
        oval +=1
        cv2.putText(img, "oval{}".format(oval),  (x, y), font, 0.8, (0))
        
    else:
        circle +=1
        cv2.putText(img, "circle{}".format(circle), (x, y), font, 0.8, (0))
        
# 結果の画像作成
cv2.imwrite('/Users/hayashidekazuyuki/Desktop/Git_Win_Air/class_CV/final_report_svm/source/IMG_8511_d.jpg',img)

# 図形の数の結果
print('Number of triangle = ' , triangle)
print('Number of rectangle = ' , rectangle)
print('Number of pentagon = ' , pentagon)
print('Number of circle = ' , circle)
print('Number of oval = ' , oval)