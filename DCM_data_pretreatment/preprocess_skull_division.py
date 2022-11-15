import argparse
import math

from cv2.cv2 import *
import numpy as np


def SkullDivision(src_image):
    """
    the function to Detach the skull for single image
    :param src_image: numpy array(1 channel in range [0,255])
    :return: image after process
    """
    img = src_image.copy()
    imshow("source", src_image)
    threshold(img, 50, 255, THRESH_BINARY, img)

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

    dst = src_image.copy()
    dst[img != 255] = 0
    imshow("out", dst)
    waitKey(0)
    return dst


def get_distance(x, y):
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return math.sqrt(dx * dx + dy * dy)


def run(src_image_path):
    src_image = imread(src_image_path, CV_8UC1)
    SkullDivision(src_image)


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="for test, written by Windbell")
    parser.add_argument("--input_path", default="./sample/sample_dcm.png", help="test dcm file")
    args = parser.parse_args()
    print(args)
    run(args.input_path)
