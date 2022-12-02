import sys
from datetime import datetime

import torch.utils.data
import torchvision
from torchvision.transforms import transforms
from torch.utils.data.dataset import T_co
import numpy as np
import time
from tqdm import tqdm

from net.ResNet50 import *
from torch.utils.tensorboard import SummaryWriter

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# class Test_MyDataset(torch.utils.data.Dataset):
#     def __init__(self, data_file, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])):
#         dataset = open(data_file, 'r')
#         data_list = []
#         for item in dataset:
#             item_list = item.split('  ')
#             data_list.append([item_list[0], int(item_list[1][0])])
#         self.data_list = data_list
#         self.trans = transform
#
#     def __getitem__(self, index) -> T_co:
#         img_path, label = self.data_list[index]
#         img = imread(img_path + "/10.png")
#         if self.trans is not None:
#             img = self.trans(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.data_list)
train_summary = SummaryWriter(log_dir="./logs/train" + current_time)
val_summary = SummaryWriter(log_dir="./logs/val" + current_time)
if __name__ == "__main__":
    net = ResNet50(ResBlock)
    # print(net)
    # net.load_state_dict(torch.load("./net/resnet50.pth"))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    print("Basic model & normalization module loaded")
    # train_data = Test_MyDataset("./data/train.txt")
    # val_data = Test_MyDataset("./data/val.txt")
    # test_data = Test_MyDataset("./data/test.txt")
    train_data = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
    val_data = torchvision.datasets.ImageFolder("dataset/val", transform=transform)
    test_data = torchvision.datasets.ImageFolder("dataset/test", transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4)
    print("datasets loaded")
    loss_func = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use device {}".format(device))
    epochs = 200

    log = open("./logs" + current_time + ".txt", 'w')
    best_accuracy = 0.0
    best_loss = 0.0
    net.to(device)
    print("start training")
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        net.train()

        train_loss = 0.0
        val_accuracy = 0.0
        val_loss = 0.0

        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        for i, (img, label) in enumerate(train_bar):
            imgs = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = net(imgs)

            loss = loss_func(outputs, label)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            # record images, loss
            grid = torchvision.utils.make_grid(imgs)
            train_summary.add_image("img", grid)
            train_summary.add_scalar("train_loss", loss.item(), i)

        with torch.no_grad():
            acc = 0.0
            net.eval()
            val_bar = tqdm(val_loader, leave=True, file=sys.stdout)
            for i, (img, label) in enumerate(val_bar):
                imgs = img.to(device)
                label = label.to(device)

                outputs = net(imgs)

                loss = loss_func(outputs, label)

                val_loss += loss.item()
                predictions = torch.max(outputs.data, dim=1)[1]
                acc += torch.eq(predictions, label).sum().item()

        train_loss = train_loss / len(train_data)
        val_accuracy = acc / len(val_data)
        val_loss = val_loss / len(val_data)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_loss = val_loss
            torch.save(net, "./logs/best.pth")

        log_info = "Training" + " Loss: %.4f\n" % train_loss \
                   + "Val Accuracy: %4.2f" % val_accuracy + " Loss: %.4f\n" % val_loss + \
                   "using time: %4.2f s\n" % (time.time() - start_time) + "best acc: %4.2f \n" % best_accuracy
        log.write(log_info)
        print(log_info)
