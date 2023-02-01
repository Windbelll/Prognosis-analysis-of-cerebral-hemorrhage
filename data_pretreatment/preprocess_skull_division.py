import argparse
import math
import os
import time

import yaml
import cv2
import numpy as np
from preprocess_format_convert import convert_dcm


def SkullDivision(src_image, m_threshold, temp=None):
    """
    the function to Detach the skull for single image
    :param src_image: numpy array(1 channel in range [0,255])
    :return: image after process
    """
    src_img = np.array(src_image, copy=True)
    _, img = cv2.threshold(src_img, m_threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("temp_grey", img)
    # cv2.morphologyEx(img, cv2.MORPH_OPEN, (3, 3), dst=img, iterations=5)
    # cv2.imshow("temp_morph", img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    object_index = 0
    max_area = 0
    # print(stats)
    for i in range(1, len(stats)):
        # distance = get_distance((centroids[i][0], centroids[i][1]), (256, 256))
        if stats[i][4] > max_area:
            max_area = stats[i][4]
            object_index = i
        else:
            continue

    img[labels != object_index] = 0
    img[labels == object_index] = 255

    cv2.imshow("temp_choose", img)

    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.fillPoly(img, contours[object_index], 255)
    _, img_2 = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow("th2", img_2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_2)
    object_index = 0
    min = 999
    for i in range(1, len(centroids)):
        if stats[i][4] <= 8000:
            continue
        distance = get_distance((centroids[i][0], centroids[i][1]), (256, 256))
        if distance < min:
            min = distance
            object_index = i
    img_2[labels != object_index] = 0
    cv2.imshow("temp", img_2)
    # cv2.fillPoly(img, contours[object_index], 255)
    img[img_2 == 255] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    object_index = 0
    max_area = 0
    # print(stats)
    for i in range(1, len(stats)):
        # distance = get_distance((centroids[i][0], centroids[i][1]), (256, 256))
        if stats[i][4] > max_area:
            max_area = stats[i][4]
            object_index = i
        else:
            continue

    img[labels != object_index] = 0
    img[labels == object_index] = 255
    # dst = np.zeros((src_image.shape[0], src_image.shape[1], 1), np.uint8)
    dst = src_img.copy()
    # ex = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # if object_index != 0:
    #     cv2.drawContours(ex, contours, object_index, (0, 255, 255), 2)
    # # cv2.imshow("111", ex)

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
        src_image = cv2.imread(file_path, cv2.CV_8UC1)
        print(src_image)
        cv2.imshow("src", src_image)
        test = SkullDivision(src_image, m_threshold)
        cv2.imshow("test_single_image", test)
        cv2.waitKey(0)
    else:
        if not os.path.exists("./output_skull"):
            os.mkdir("./output_skull")
        files = os.listdir(file_path)
        for patient_name in files:
            # if not os.path.exists("../data/good/" + patient_name):
            #     continue
            index = 0
            os.mkdir("./output_skull/" + patient_name)
            datas = os.listdir(file_path + "/" + patient_name)
            total = len(datas)
            for data in datas:
                index += 1
                start_time = time.perf_counter()
                path = file_path + "/" + patient_name + "/" + data
                # source_path = "../data/good/" + patient_name + "/" + data
                src = cv2.imread(path, cv2.CV_8UC1)
                # temp = cv2.imread(source_path, cv2.CV_8UC1)
                dst = SkullDivision(src, m_threshold)
                cv2.imwrite("./output_skull/" + patient_name + "/" + data, dst)
                print("process: " + patient_name + "(%d/%d)" % (index, total),
                      "(%.2fms)" % ((time.perf_counter() - start_time) * 1000))
        print("complete!")


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for dataset, written by Windbell")
    parser.add_argument("--input_path", default="../data/goodnewsrc", help="dataset png file or the dir in ("
                                                                    "here)/patient_name ... etc")
    parser.add_argument("--as_dir", default=True, help="process all image in a dir")
    parser.add_argument("--threshold", default=50, help="the threshold to seg image")
    parser.add_argument('-y', '--yaml', default=False, help="use yaml to load hyps")
    parser.add_argument('--yaml_path', default="settings/normal.yaml", help="yaml file")
    args = parser.parse_args()
    print(args)
    if args.as_dir:
        if args.yaml:
            with open(args.yaml_path, "r") as ff:
                setting = yaml.load(ff, Loader=yaml.FullLoader)
            run(args.input_path, 1, setting['threshold'])
        else:
            run(args.input_path, 1, args.threshold)
    else:
        if args.yaml:
            with open(args.yaml_path, "r") as ff:
                setting = yaml.load(ff, Loader=yaml.FullLoader)
            run(args.input_path, 0, setting['threshold'])
        else:
            run(args.input_path, 0, args.threshold)
