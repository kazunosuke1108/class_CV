import numpy as np
import cv2

def dist_pt_line(pt,line_pt1,line_pt2):
    vec_u=np.array(pt)-np.array(line_pt1)
    print(vec_u)
    vec_v=np.array(line_pt2)-np.array(line_pt1)
    print(vec_v)
    distance=np.linalg.norm(vec_u)*np.sin(np.arccos(np.inner(vec_u,vec_v)/np.linalg.norm(vec_u)/np.linalg.norm(vec_v)))
    print(distance)
    return distance

img1=np.zeros((500,500))

x0,y0=100,100
x1,y1=400,400
img1 = cv2.line(img1, (x0,y0), (x1,y1), (255,255,255),1)

pt=(300,200)

distance=dist_pt_line(pt,(x0,y0), (x1,y1))
img1=cv2.circle(img1,pt,10,(255,255,255))
img1=cv2.circle(img1,pt,int(distance),(255,255,255))

cv2.imshow("img1",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()