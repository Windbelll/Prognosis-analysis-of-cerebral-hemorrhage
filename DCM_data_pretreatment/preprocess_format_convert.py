import argparse

import pydicom
import os
import numpy as np
import png
import time

import yaml


def ConvertAllDCMs(file_path, width, level):
    """
    process all dcm files in dataset dir, support format "data_root/*/patient_name"
    :return: no return, write png images in "./output/patient1*/...*"
    """
    if not os.path.exists("./output"):
        os.mkdir("./output")
    names_with_gcs = os.listdir(file_path)
    for name in names_with_gcs:
        all_image_names = []
        output_name = name[:12].replace(' ','_')
        # cd to dcm dir
        temp_path = file_path + "/" + name
        temp_names = os.listdir(temp_path)
        if len(temp_names) == 1:
            temp_path = temp_path + "/" + temp_names[0]
            temp_names = os.listdir(temp_path)
            if len(temp_names) == 1:
                temp_path = temp_path + "/" + temp_names[0]

        # pick all images
        names = os.listdir(temp_path)
        if len(names) == 0:
            continue
        if not os.path.exists("./output/" + output_name):
            os.mkdir("./output/" + output_name)

        for image_name in names:
            index = image_name.rfind('.')
            image_name = image_name[:index]
            all_image_names.append(image_name)
        # convert module
        index = 0
        total = len(all_image_names)
        for image_name in all_image_names:
            index += 1
            start_time = time.perf_counter()
            # set src/dst path
            src_path = temp_path + "/" + image_name + ".dcm"
            dst_path = "./output/" + output_name + "/" + str(index) + ".png"

            dst = convert_dcm(src_path, width, level)
            w = png.Writer(dst.shape[1], dst.shape[0], greyscale=True)
            object_file = open(dst_path, 'wb')
            w.write(object_file, dst)
            print("process: " + name + "(%d/%d)" % (index, total),
                  "(%.2fms)" % ((time.perf_counter() - start_time) * 1000))


def convert_dcm(src_path, width, level):
    """
    process single dcm file
    """
    # read file & use W/L Algorithm to convert color space
    src_file = pydicom.read_file(src_path)
    src = src_file.pixel_array
    src = src.astype(np.int16)

    if width == 0:
        src = np.int8(src)
        return src
    intercept = src_file.RescaleIntercept
    slope = src_file.RescaleSlope
    if slope != 1:
        src = slope * src.astype(np.float64)
        src = src.astype(np.int16)

    src += np.int16(intercept)

    window_level = level
    window_width = width
    min_val = (2 * window_level - window_width) / 2.0 + 0.5
    max_val = (2 * window_level + window_width) / 2.0 + 0.5
    dst_image = (src - min_val) * 255.0 / float(max_val - min_val)
    dst_image = np.uint8(dst_image)
    return dst_image


if __name__ == '__main__':
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for test, written by Windbell")
    parser.add_argument("--input_path", default="C:/Users/49804/Desktop/文章/大三/创新实践/预后不良/预后不良",
                        help="the dir to (here)/patient_name_gcs_etc")
    parser.add_argument('-y', '--yaml', default=True, help="use yaml to load hyps")
    parser.add_argument("--window_level", type=int, default=50, help="the level in W/L algorithm in CT")
    parser.add_argument("--window_width", type=int, default=100, help="the width in W/L algorithm in CT")
    args = parser.parse_args()
    print(args)
    if args.yaml:
        with open("./settings/test.yaml", "r") as ff:
            setting = yaml.load(ff, Loader=yaml.FullLoader)
        print("use yaml: window_width = " + str(setting['window_width']) + "   window_level = "
              + str(setting['window_level']))
        ConvertAllDCMs(args.input_path, int(setting['window_width']), int(setting['window_level']))
    else:
        ConvertAllDCMs(args.input_path, args.window_width, args.window_level)
