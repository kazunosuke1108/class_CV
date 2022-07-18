import glob
import os
import matplotlib.pyplot as plt
import cv2





os.chdir("final_report_SfM")
current_dir = os.getcwd()
"""
"""
interpolated_images=sorted(glob.glob(current_dir+"/results/view_interpolation/*"))
interpolated_nos=[]
for interpolated_image in interpolated_images:
    interpolated_no=os.path.basename(interpolated_image)[:5]
    interpolated_nos.append(interpolated_no)

original_images=sorted(glob.glob(current_dir+"/images/images3/*"))

for original_image in original_images:
    original_no=os.path.basename(original_image)[:-4]
    try:
        interpolated_no=interpolated_nos[0]
        interpolated_image=interpolated_image[0]
    except IndexError:
        print(original_no,"XXXXX","NO MATCH (interpolated empty)")
        break
    if original_no==interpolated_no:
        print(original_no,interpolated_no,"PERFECT MATCH")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
    elif int(original_no)<int(interpolated_no):
        print(original_no,"XXXXX","NO MATCH (INTERPOLATION FAILED)")
    elif int(original_no)>int(interpolated_no):
        print(original_no,interpolated_no,"INTERPOLATION MATCH")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
    else:
        print("### ERROR ###")

fig=plt.figure(figsize=(7,3))
path1=current_dir+"/images/images3/00300.png"
path2=current_dir+"/results/view_interpolation/00300_1-00.jpg"
img1=cv2.imread(path1)
ax1=fig.add_subplot(1,2,1)
ax1.set_title("original image")
ax1.axis("off")
plt.imshow(img1)
img2=cv2.imread(path2)
ax2=fig.add_subplot(1,2,2)
ax2.set_title("interpolated image")
ax2.axis("off")
plt.imshow(img2)
plt.savefig(current_dir+"/results/comparison_movie/test.png")
