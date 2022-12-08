import os
import numpy as np
from cv2 import imread, imwrite

# settings total = 100
train_rate = 80
val_rate = 10
test_rate = 10


def start():
    if not os.path.exists("./bad"):
        print("error: no bad data")
    if not os.path.exists("./good"):
        print("error: no good data")
    train = open("./train.txt", 'w')
    val = open("./val.txt", 'w')
    test = open("./test.txt", 'w')

    bad_names = os.listdir("./bad")
    for name in bad_names:
        choice = np.random.randint(100)
        if choice < train_rate:
            train.write("bad/" + name + " " + str(0) + "\n")
        elif choice < train_rate + val_rate:
            val.write("bad/" + name + " " + str(0) + "\n")
        else:
            test.write("bad/" + name + " " + str(0) + "\n")

    good_names = os.listdir("./good")
    for name in good_names:
        choice = np.random.randint(100)
        if choice < train_rate:
            train.write("good/" + name + " " + str(1) + "\n")
        elif choice < train_rate + val_rate:
            val.write("good/" + name + " " + str(1) + "\n")
        else:
            test.write("good/" + name + " " + str(1) + "\n")


def load_dataset():
    train = open("./train.txt", 'r')
    os.mkdir("../dataset/train")
    os.mkdir("../dataset/train/bad")
    os.mkdir("../dataset/train/good")
    for item in train:
        item_list = item.split(" ")
        nums = len(os.listdir(item_list[0]))
        for i in range(5):
            img = imread(item_list[0] + "/%d.png" % (int(nums / 2) + (2 - i)))
            imwrite("../dataset/train/" + item_list[0] + "-%d.png" % i, img)

    val = open("./val.txt", 'r')
    os.mkdir("../dataset/val")
    os.mkdir("../dataset/val/bad")
    os.mkdir("../dataset/val/good")
    for item in val:
        item_list = item.split(" ")
        nums = len(os.listdir(item_list[0]))
        for i in range(5):
            img = imread(item_list[0] + "/%d.png" % (int(nums / 2) + (2 - i)))
            imwrite("../dataset/val/" + item_list[0] + "-%d.png" % i, img)

    test = open("./test.txt", 'r')
    os.mkdir("../dataset/test")
    os.mkdir("../dataset/test/bad")
    os.mkdir("../dataset/test/good")
    for item in test:
        item_list = item.split(" ")
        nums = len(os.listdir(item_list[0]))
        for i in range(5):
            img = imread(item_list[0] + "/%d.png" % (int(nums / 2) + (2 - i)))
            imwrite("../dataset/test/" + item_list[0] + "-%d.png" % i, img)


if __name__ == "__main__":
    start()
    load_dataset()
