import torch.utils.data
import torchvision
from torchvision.transforms import transforms
from torch.utils.data.dataset import T_co
import numpy as np
import time

from net.ResNet50 import *
from cv2 import imread


class Test_MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])):
        dataset = open(data_file, 'r')
        data_list = []
        for item in dataset:
            item_list = item.split('  ')
            data_list.append([item_list[0], int(item_list[1][0])])
        self.data_list = data_list
        self.trans = transform

    def __getitem__(self, index) -> T_co:
        img_path, label = self.data_list[index]
        img = imread(img_path + "/10.png")
        if self.trans is not None:
            img = self.trans(img)
        return img, label

    def __len__(self):
        return len(self.data_list)


def train_model(model, loss_func, optimizer, device,epochs=50):
    log = open("./logs/log.txt", 'w')
    best_accuracy = 0.0
    best_loss = 0.0
    model.to(device)
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_accuracy = 0.0
        train_loss = 0.0
        val_accuracy = 0.0
        val_loss = 0.0

        for i, (img, label) in enumerate(train_data):
            imgs = torch.Tensor(np.expand_dims(img, 0))
            imgs = imgs.to(device)
            label = torch.Tensor(np.expand_dims(np.array(label), 0))
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = loss_func(outputs, label.long())
            loss.backward()

            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_num = predictions.eq(label.data.view_as(predictions))

            acc = torch.mean(correct_num.type(torch.FloatTensor))
            train_accuracy += acc.item() * imgs.size(0)
            print("train check!")

        with torch.no_grad():
            model.eval()
        for i, (img, label) in enumerate(val_data):
            imgs = torch.Tensor(np.expand_dims(img, 0))
            imgs = imgs.to(device)
            label = torch.Tensor(np.expand_dims(np.array(label), 0))
            label = label.to(device)

            outputs = model(imgs)

            loss = loss_func(outputs, label.long())

            val_loss += loss.item() * imgs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_num = predictions.eq(label.data.view_as(predictions))

            acc = torch.mean(correct_num.type(torch.FloatTensor))
            val_accuracy += acc.item() * imgs.size(0)
        train_accuracy = train_accuracy / len(train_data)
        train_loss = train_loss / len(train_data)
        val_accuracy = val_accuracy / len(val_data)
        val_loss = val_loss / len(val_data)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_loss = val_loss
            torch.save(model, "./logs/best.pth")

        log_info = "Epoch: %3d\n" % epoch + " Training Accuracy: %4.2f" % train_accuracy + " Loss: %.4f\n" % train_loss\
                   + "val Accuracy: %4.2f" % val_accuracy + " Loss: %.4f\n" % val_loss + \
                   "using time: %4d s\n" % ((time.time() - start_time) / 1000) + "best acc:%4.2f \n" % best_accuracy
        log.write(log_info)
        print(log_info)


if __name__ == "__main__":
    net = ResNet50(ResBlock)
    # print(net)
    # net.load_state_dict(torch.load("./net/resnet50.pth"))

    train_data = Test_MyDataset("./data/train.txt")
    val_data = Test_MyDataset("./data/val.txt")
    test_data = Test_MyDataset("./data/test.txt")

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(net, loss_func, optimizer, device, 2)
