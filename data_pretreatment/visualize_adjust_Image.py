import argparse

import numpy as np
import yaml
import numpy
from cv2.cv2 import *
from preprocess_format_convert import convert_dcm
from preprocess_skull_division import SkullDivision


def callback(value):
    pass


def run(file_path, yaml_path):
    namedWindow("preview", WINDOW_AUTOSIZE)
    namedWindow("source", WINDOW_AUTOSIZE)
    namedWindow("division", WINDOW_AUTOSIZE)
    src = convert_dcm(file_path, 0, 0)
    imshow("source", src)
    imshow("preview", src)
    imshow("division", src)

    with open("./settings/default.yaml", "r") as ff:
        setting = yaml.load(ff, Loader=yaml.FullLoader)

    createTrackbar("window_width", "preview", setting['window_width'], 500, callback)
    createTrackbar("window_level", "preview", setting['window_level'], 900, callback)
    createTrackbar("threshold", "division", setting['threshold'], 255, callback)
    while True:
        width = getTrackbarPos("window_width", "preview")
        # print(width)
        level = getTrackbarPos("window_level", "preview")
        threshold_m = int(getTrackbarPos("threshold", "division"))
        img = convert_dcm(file_path, width, level)
        imshow("preview", img)
        src = np.array(img, copy=True)
        dst = SkullDivision(src, threshold_m)
        imshow("division", dst)
        key = waitKey(10)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            imwrite("./sample/sample_dcm.png", img)
            setting['window_width'] = width
            setting['window_level'] = level
            setting['threshold'] = threshold_m
            with open(yaml_path, "w") as fw:
                fw.write(yaml.dump(setting))


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for dataset, visualizer, written by Windbell")
    parser.add_argument("--input_path", default="./sample/sample3.dcm", help="dataset dcm file")
    parser.add_argument("--output_yaml_path", default="./settings/test.yaml", help="dataset dcm file")
    args = parser.parse_args()
    print(args)
    run(args.input_path, args.output_yaml_path)
