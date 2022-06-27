from PIL import Image
import pyheif
import os

def conv(image_path):
    new_name = image_path.replace('HEIC', 'png')
    heif_file = pyheif.read(image_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    os.remove(image_path)
    data.save(new_name, "png")

import glob
lst = sorted(glob.glob("/home/ytpc2019a/code_ws/class_CV/final_report_SfM/img_calib_normal/*.HEIC"))
for l in lst:
    conv(l)