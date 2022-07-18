import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

os.chdir("final_report_SfM")
current_dir = os.getcwd()

def crt_results(path1,path2,title1="original image",title2="interpolated image",color=""):
    fig=plt.figure(figsize=(7,3))
    img1=cv2.imread(path1)
    ax1=fig.add_subplot(1,2,1)
    ax1.set_title(title1)
    ax1.axis("off")
    plt.imshow(img1)
    if path2==None:
        img2=np.zeros_like(img1)
    else:
        img2=cv2.imread(path2)
    ax2=fig.add_subplot(1,2,2)
    if color!="":
        ax2.set_title(title2,color=color)
    else:
        ax2.set_title(title2)
    ax2.axis("off")
    plt.imshow(img2)
    plt.savefig(current_dir+"/results/comparison_movie/test.png")
    pass


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
        interpolated_image=interpolated_images[0]
    except IndexError:
        print(original_no,"XXXXX","NO MATCH (interpolated empty)")
        break
    if original_no==interpolated_no:
        print(original_no,interpolated_no,"PERFECT MATCH")
        crt_results(original_image,interpolated_image,title1=f"original no.{original_no}",title2=f"original no.{interpolated_no}")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
    elif int(original_no)<int(interpolated_no):
        crt_results(original_image,None,title1=f"original no.{original_no}",title2=f"interp. (failed) no.{interpolated_no}")
        print(original_no,"XXXXX","NO MATCH (INTERPOLATION FAILED)")
    elif int(original_no)>int(interpolated_no):
        crt_results(original_image,interpolated_image,title1=f"original no.{original_no}",title2=f"interp. no.{interpolated_no}",color="red")
        print(original_no,interpolated_no,"INTERPOLATION MATCH")
        interpolated_nos=interpolated_nos[1:]
        interpolated_images=interpolated_images[1:]
    else:
        print("### ERROR ###")


