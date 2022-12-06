import sys
import time
import torch
from datetime import datetime

import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())


def train(net, device, epochs, train_loader, val_loader, optimizer, loss_func):
    """
    This is the function of training module
    """
    train_summary = SummaryWriter(log_dir="./logs/train" + current_time)
    val_summary = SummaryWriter(log_dir="./logs/val" + current_time)
    log = open("./logs/local" + current_time + ".txt", 'w')
    best_accuracy = 0.0
    best_epoch = 0
    net.to(device)
    print("start training")
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        net.train()

        train_loss = 0.0
        train_count = 0

        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        for i, (img, label) in enumerate(train_bar):
            train_count += 1
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

        val_accuracy, val_loss = val_without_info(net, val_loader, device, loss_func)
        train_loss = train_loss / train_count
        train_summary.add_scalar("train_loss", train_loss, epoch)
        val_summary.add_scalar("val_acc", val_accuracy, epoch)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(net, "./logs/best.pth")

        log_info = "-------\n" + "Epoch %d:\n" % (epoch + 1) + "Training" + " Loss: %.4f\n" % train_loss \
                   + "Val Accuracy: %4.2f" % val_accuracy + " Loss: %.4f\n" % val_loss + \
                   "using time: %4.2f s\n" % (time.time() - start_time) + "best acc: %4.2f" % best_accuracy \
                   + " produced @epoch %3d\n" % (best_epoch + 1)
        log.write(log_info)
        print(log_info)


def val_without_info(net, val_loader, device, loss_func):
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        acc = 0.0
        net.eval()
        val_bar = tqdm(val_loader, leave=True, file=sys.stdout)
        for i, (img, label) in enumerate(val_bar):
            val_count += 1
            # 5张投票 / 单张
            imgs = img.to(device)
            label = label.to(device)

            outputs = net(imgs)

            loss = loss_func(outputs, label)

            val_loss += loss.item()
            predictions = torch.max(outputs.data, dim=1)[1]
            acc += torch.eq(predictions, label).sum().item()
        val_accuracy = acc / val_count
        val_loss = val_loss / val_count
        return val_accuracy, val_loss
