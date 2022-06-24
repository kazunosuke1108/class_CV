import cv2
import os
import shutil
import glob

save_dir='class_CV/final_report/kabe_images'

def capture(device_num, delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        print(f"camera no.{device_num} not found")
        return
    n = 0
    try:
        shutil.rmtree(save_dir)
    except FileNotFoundError:
        pass
    os.mkdir(save_dir)
    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        n += 1
        if n%10==0:#key == ord('c'):
            cv2.imwrite(f'class_CV/final_report/kabe_images/{str(int(n/10)).zfill(4)}.png', frame)
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)


capture(0,)

files=glob.glob(save_dir+"/*")
for file in files:
    print(file)