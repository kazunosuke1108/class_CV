import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def moving_average(x, num=50):
    ave_data = np.convolve(x, np.ones(num)/num,mode="valid")
    return ave_data

os.chdir("final_report_SfM")
current_dir = os.getcwd()

frame_list=[]
result_list=[]#failed=0,interpolated=0.5,original=1

interpolated_images=sorted(glob.glob(current_dir+"/results/view_interpolation/*"))
interpolated_nos=[]
for interpolated_image in interpolated_images:
    interpolated_no=os.path.basename(interpolated_image)[:5]
    interpolated_nos.append(interpolated_no)

original_images=sorted(glob.glob(current_dir+"/images/images3/*"))

for original_image in original_images:
    original_no=os.path.basename(original_image)[:-4]
    frame_list.append(int(original_no))
    try:
        interpolated_no=interpolated_nos[0]
        interpolated_image=interpolated_images[0]
        savepath=current_dir+f"/results/comparison_images/{original_no}_{os.path.basename(interpolated_image)}"
    except IndexError:
        print(original_no,"XXXXX","NO MATCH (interpolated empty)")
        result_list.append(0)
        break
    if original_no==interpolated_no:
        print(original_no,interpolated_no,"PERFECT MATCH")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
        result_list.append(1)
    elif int(original_no)<int(interpolated_no):
        print(original_no,"XXXXX","NO MATCH (INTERPOLATION FAILED)")
        result_list.append(0)
    elif int(original_no)>int(interpolated_no):
        print(original_no,interpolated_no,"INTERPOLATION MATCH")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
        result_list.append(0.5)
    else:
        print("### ERROR ###")

plt.plot(frame_list,result_list)
average=moving_average(result_list)
plt.plot(frame_list[len(frame_list)-len(average):],average,label="moving average")
plt.xlabel("frame")
plt.ylabel("fail processing=0, interp=0.5, match=1")
plt.legend()
plt.savefig(current_dir+"/results/frames_overview.jpg")