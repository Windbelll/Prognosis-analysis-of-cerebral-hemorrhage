import argparse

import pydicom
import os
import numpy as np
import png


# special example in here
# 预后良好/不良 <- file path
#   -name gcs
#       -numbers
#           -nums
#               -dcm1
#               -dcm2 ...

def ConvertAlldcms(file_path, width, level):
    if not os.path.exists("./output"):
        os.mkdir("./output")
    names_with_gcs = os.listdir(file_path)
    for name in names_with_gcs:
        all_image_names = []
        # 进入dcm目录
        temp_path = file_path + "/" + name
        temp_names = os.listdir(temp_path)
        if len(temp_names) == 1:
            temp_path = temp_path + "/" + temp_names[0]
            temp_names = os.listdir(temp_path)
            if len(temp_names) == 1:
                temp_path = temp_path + "/" + temp_names[0]
        # 取图片
        names = os.listdir(temp_path)
        if len(names) == 0:
            continue
        if not os.path.exists("./output/" + name):
            os.mkdir("./output/" + name)
        for image_name in names:
            index = image_name.rfind('.')
            image_name = image_name[:index]
            all_image_names.append(image_name)
        for image_name in all_image_names:
            src_path = temp_path + "/" + image_name + ".dcm"
            dst_path = "./output/" + name + "/" + image_name + ".png"
            dst = pydicom.read_file(src_path)
            shape = dst.pixel_array.shape
            src = dst.pixel_array
            window_level = level
            window_width = width
            min_val = (2 * window_level - window_width) / 2.0 + 0.5
            max_val = (2 * window_level + window_width) / 2.0 + 0.5
            dst = (src - min_val) * 255.0 / float(max_val - min_val)
            dst = np.int8(src)
            w = png.Writer(shape[1], shape[0], greyscale=True)
            object_file = open(dst_path, 'wb')
            w.write(object_file, dst)

        exit(0)


if __name__ == '__main__':
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for test, written by windbell")
    parser.add_argument("--input_path", default="C:/Users/49804/Desktop/文章/大三/创新实践/预后不良/预后不良",help="the dir to (here)/patient_name_gcs_etc")
    parser.add_argument("--window_level", type=int, default=50, help="the level in W/L algorithm in CT")
    parser.add_argument("--window_width", type=int, default=100, help="the level in W/L algorithm in CT")
    args = parser.parse_args()
    print(args)
    ConvertAlldcms(args.input_path, args.window_width, args.window_level)