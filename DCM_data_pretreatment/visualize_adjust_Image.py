import argparse

import numpy as np
from cv2.cv2 import *
from preprocess_format_convert import convert_dcm


def callback(value):
    pass


def run(file_path):
    namedWindow("preview", WINDOW_AUTOSIZE)
    namedWindow("source", WINDOW_AUTOSIZE)
    src = convert_dcm(file_path, 0, 0)
    imshow("source", src)
    imshow("preview", src)
    createTrackbar("window_width", "preview", 0, 200, callback)
    createTrackbar("window_level", "preview", 0, 200, callback)
    while True:
        width = getTrackbarPos("window_width", "preview")
        level = getTrackbarPos("window_level", "preview")
        img = convert_dcm(file_path, width, level)
        imshow("preview", img)
        key = waitKey(10)
        if key & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for test, visualizer, written by Windbell")
    parser.add_argument("--input_path", default="./sample/sample_dcm.dcm", help="test dcm file")
    args = parser.parse_args()
    print(args)
    run(args.input_path)
