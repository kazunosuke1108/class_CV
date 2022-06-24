import cv2
import numpy as np
array=cv2.imread('/home/ytpc2019a/code_ws/class_CV/final_report/canvas.jpg',0)
print(array.shape)
flg=False
for j in range(array.shape[1]):
    for i in range(array.shape[0]):
        if i%10==0 and j%10==0:
            print(i,j,array[i,j])
        if int(array[i,j])<int(np.max(array)):
            up=i
            left=j
            flg=True
            break
    if flg:
        break

flg=False
for j in reversed(range(array.shape[1])):
    for i in reversed(range(array.shape[0])):
        if i%10==0 and j%10==0:
            print(i,j)
        if array[i,j]<np.max(array):
            down=i
            right=j
            flg=True
            break
    if flg:
        break
print(up,down,left,right)
cv2.imwrite("class_CV/final_report/cut.png",array[up:down][left:right])
print(up,down,left,right)


#cv2.imshow("cut",array[up:down][left:right])
cv2.waitKey(0)
cv2.destroyAllWindows()
