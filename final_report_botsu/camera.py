import cv2

cap1 = cv2.VideoCapture(0)
print(cap1.isOpened())
# cap2 = cv2.VideoCapture(2)
# print(cap2.isOpened())

while(1):
    ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()
    if ret1 :#and ret2:
        cv2.imshow("frame1",frame1)
        # cv2.imshow("frame2",frame2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap1.release()
# cap2.release()