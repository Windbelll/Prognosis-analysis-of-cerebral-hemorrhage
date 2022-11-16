import argparse
import math
import os
import time

import yaml
from cv2.cv2 import *
import numpy as np


def SkullDivision(src_image, m_threshold):
    """
    the function to Detach the skull for single image
    :param src_image: numpy array(1 channel in range [0,255])
    :return: image after process
    """
    src_img = np.array(src_image, copy=True)
    _, img = threshold(src_img, m_threshold, 255, THRESH_BINARY)

    morphologyEx(img, MORPH_OPEN, (7, 7), dst=img, iterations=5)

    num_labels, labels, stats, centroids = connectedComponentsWithStats(img)
    object_index = 0
    min_distance = 255
    for i in range(len(centroids)):
        distance = get_distance((centroids[i][0], centroids[i][1]), (256, 256))
        if distance < min_distance:
            min_distance = distance
            object_index = i
        else:
            continue

    img[labels != object_index] = 0
    img[labels == object_index] = 255

    contours, hierarchy = findContours(img, RETR_TREE, CHAIN_APPROX_NONE)

    for i in range(1, len(contours)):
        img = fillPoly(img, [contours[i]], 255)

    # dst = np.zeros((src_image.shape[0], src_image.shape[1], 1), np.uint8)
    dst = src_image.copy()
    dst[img != 255] = 0
    return dst


def get_distance(x, y):
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return math.sqrt(dx * dx + dy * dy)


def run(file_path, mode, m_threshold):
    """
    mode 0: single image
    mode 1: dir
    """
    if mode == 0:
        src_image = imread(file_path, CV_8UC1)
        print(src_image)
        SkullDivision(src_image, m_threshold)
    else:
        if not os.path.exists("./output_skull"):
            os.mkdir("./output_skull")
        files = os.listdir(file_path)
        for patient_name in files:
            index = 0
            os.mkdir("./output_skull/" + patient_name)
            datas = os.listdir(file_path + "/" + patient_name)
            total = len(datas)
            for data in datas:
                index += 1
                start_time = time.perf_counter()
                path = file_path + "/" + patient_name + "/" + data
                src = imread(path, CV_8UC1)
                dst = SkullDivision(src, m_threshold)
                imwrite("./output_skull/" + patient_name + "/" + data, dst)
                print("process: " + patient_name + "(%d/%d)" % (index, total),
                      "(%.2fms)" % ((time.perf_counter() - start_time) * 1000))
            waitKey(0)
        print("complete!")


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for test, written by Windbell")
    parser.add_argument("--input_path", default="./output", help="test dcm file or the dir in ("
                                                                                "here)/patient_name ... etc")
    parser.add_argument("--as_dir", default=True, help="process all image in a dir")
    parser.add_argument("--threshold", default=60, help="the threshold to seg image")
    parser.add_argument('-y', '--yaml', default=True, help="use yaml to load hyps")
    args = parser.parse_args()
    print(args)
    if args.as_dir:
        if args.yaml:
            with open("./settings/test.yaml", "r") as ff:
                setting = yaml.load(ff, Loader=yaml.FullLoader)
            run(args.input_path, 1, setting['threshold'])
        else:
            run(args.input_path, 1, args.threshold)
    else:
        if args.yaml:
            with open("./settings/test.yaml", "r") as ff:
                setting = yaml.load(ff, Loader=yaml.FullLoader)
            run(args.input_path, 0, setting['threshold'])
        else:
            run(args.input_path, 0, args.threshold)
