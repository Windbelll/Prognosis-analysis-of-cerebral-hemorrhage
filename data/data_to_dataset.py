import os

import cv2
import numpy
import numpy as np
import torch.utils.data
from cv2 import imread, imwrite

# settings total = 100
from torchvision.transforms import transforms

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
            # img = add_gcs(img, item_list[0])
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
            # img = add_gcs(img, item_list[0])
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
            # img = add_gcs(img, item_list[0])
            imwrite("../dataset/test/" + item_list[0] + "-%d.png" % i, img)


def add_gcs(img: np.ndarray, item: str) -> np.ndarray:
    item_list = item.split('_')
    gcs = int(item_list[len(item_list) - 1])
    img[img < 15] = (15 - gcs) * (240 / 15) + 15
    return img


def get_gcs(item: str) -> int:
    item_list = item.split('_')
    gcs = int(item_list[len(item_list) - 1])
    return gcs


if __name__ == "__main__":
    # start()
    load_dataset()


class gcs_dataset(torch.utils.data.Dataset):
    def __init__(self, data_list: str = None):
        self.data_list = []
        self.list = open(data_list, 'r')
        for item in self.list:
            item_list = item.split(" ")
            name = item_list[0]
            nums = len(os.listdir("data/" + item_list[0]))
            name_list = name.split('_')
            gcs = int(name_list[len(name_list) - 1])
            label = int(item_list[1])
            for i in range(5):
                single_name = name + "/%d.png" % (int(nums / 2) + (2 - i))
                self.data_list.append([single_name, gcs, label])

    def __getitem__(self, index):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((512, 512)),
             transforms.Normalize([0.1747, 0.1747, 0.1747], [0.3030, 0.3030, 0.3030])])
        # debug_patient_name = self.data_list[index][0]
        img = cv2.imread("data/" + self.data_list[index][0], 0)
        # img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = transform(img)
        gcs_step = self.data_list[index][1]
        gcs_np = []
        for i in range(15):
            gcs_step -= 1
            if gcs_step >= 0:
                gcs_np.append(1)
            else:
                gcs_np.append(0)
        label = self.data_list[index][2]
        gcs_tensor = torch.FloatTensor(numpy.array(gcs_np))
        label_tensor = torch.from_numpy(numpy.array(label)).long()

        return img, gcs_tensor, label_tensor

    def __len__(self):
        return len(self.data_list)


def get_mean_std(loader):
    """
    calculate mean and std, tips: the loader should only include images(type: numpy, unsigned int8)
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in loader:
        # print(data)
        channels_sum += torch.mean(data.float().div(255))
        # print(channels_sum.shape)#torch.Size([3])
        channels_squared_sum += torch.mean(data.float().div(255) ** 2)
        # print(channels_squared_sum.shape)#torch.Size([3])
        num_batches += 1

    e_x = channels_sum / num_batches
    e_x_squared = channels_squared_sum / num_batches
    var = e_x_squared - e_x ** 2

    return e_x, var ** 0.5
