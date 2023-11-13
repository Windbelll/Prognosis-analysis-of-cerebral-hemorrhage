import os
import sys
import time

import cv2
import numpy

import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())


def run_gcs(resnet, gcsnet, self_attn, classify, device, epochs, train_loader, val_loader, optimizer, loss_func,
            train_c, val_c, with_gcs, target_gcs=False):
    """
    This is the function of training module
    """
    if with_gcs and target_gcs:
        raise ValueError("can't use both target and input gcs score!")
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    os.mkdir("./logs/" + current_time)
    train_summary = SummaryWriter(log_dir="./logs/" + current_time + "/summary")
    log = open("./logs/" + current_time + "/training_log.txt", 'w')

    best_accuracy = best_acc_batch = 0.0
    best_epoch = best_epoch_batch = 0
    # load models to cuda or cpu
    resnet = resnet.to(device)
    classify = classify.to(device)
    if with_gcs:
        gcsnet = gcsnet.to(device)
        self_attn = self_attn.to(device)

    print("start training")
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        resnet.train()
        classify.train()
        if with_gcs:
            gcsnet.train()
            self_attn.train()

        train_loss = 0.0
        train_count = train_c

        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
        for i, (img, gcs, label) in enumerate(train_bar):
            img = img.to(device)
            gcs = gcs.to(device)
            label = label.to(device)
            print(img.size())
            optimizer.zero_grad()
            global_feature = resnet(img)
            # if use gcs to be input, we will mix features by multi-head self-attention module, else classify the image
            # feature straightly(notice the dim transformation)
            if with_gcs:
                gcs_feature = gcsnet(gcs)
                print(global_feature.size())
                print(gcs_feature.size())
                global_feature = torch.cat([global_feature, gcs_feature], dim=1)
                global_feature = torch.unsqueeze(global_feature, 1)
                print(global_feature.size())
                global_feature = self_attn(global_feature)
            else:
                # global_feature = torch.unsqueeze(global_feature, 1)
                global_feature = global_feature

            # outputs = classify(global_feature)
            outputs = global_feature
            loss = loss_func(outputs, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        val_loss = 0.0
        val_count = val_c
        with torch.no_grad():
            acc = 0.0
            resnet.eval()
            # classify.eval()
            if with_gcs:
                gcsnet.eval()
                self_attn.eval()
            val_bar = tqdm(val_loader, leave=True, file=sys.stdout)
            for i, (img, gcs, label) in enumerate(val_bar):
                img = img.to(device)
                label = label.to(device)
                gcs = gcs.to(device)

                global_feature = resnet(img)

                if with_gcs:
                    gcs_feature = gcsnet(gcs)
                    global_feature = torch.cat([global_feature, gcs_feature], dim=1)
                    global_feature = torch.unsqueeze(global_feature, 1)
                    global_feature = self_attn(global_feature)
                else:
                    # global_feature = torch.unsqueeze(global_feature, 1)
                    global_feature = global_feature
                # outputs = classify(global_feature)
                outputs = global_feature
                loss = loss_func(outputs, label)

                val_loss += loss.item()
                predictions = torch.max(outputs.data, dim=1)[1]
                acc += torch.eq(predictions, label).sum().item()

            val_accuracy = acc / val_count
            val_loss = val_loss / val_count
        vals_accuracy, _ = val_batch_gcs(resnet=resnet, gcsnet=gcsnet, classifier=classify, loss_func=loss_func,
                                         self_attn=self_attn, device=device, with_gcs=with_gcs)
        train_loss = train_loss / train_count

        train_summary.add_scalar("train_loss", train_loss, epoch)
        train_summary.add_scalar("val_acc", val_accuracy, epoch)
        train_summary.add_scalar("val_s_acc", vals_accuracy, epoch)

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(resnet, "./logs/" + current_time + "/best_every_res.pth")
            # torch.save(classify, "./logs/" + current_time + "/best_every_dns.pth")
            if with_gcs:
                torch.save(gcsnet, "./logs/" + current_time + "/best_every_gcs.pth")
                torch.save(self_attn, "./logs/" + current_time + "/best_every_attn.pth")
        if vals_accuracy >= best_acc_batch:
            best_acc_batch = vals_accuracy
            best_epoch_batch = epoch
        #     torch.save(net, "./logs/best_batch.pth")
        log_info = "-------\n" + "Epoch %d:\n" % (epoch + 1) + "Training" + " Loss: %.4f\n" % train_loss \
                   + "Val Accuracy: %4.2f%% @every || " % (val_accuracy * 100) + \
                   "%4.2f%% @batch" % (vals_accuracy * 100) + \
                   " || Loss: %.4f\n" % val_loss + \
                   "using time: %4.2f s\n" % (time.time() - start_time) + \
                   "best every acc: %4.2f%%" % (best_accuracy * 100) \
                   + " produced @epoch %3d\n" % (best_epoch + 1) + \
                   "best batch acc: %4.2f%%" % (best_acc_batch * 100) \
                   + " produced @epoch %3d\n" % (best_epoch_batch + 1)
        log.write(log_info)
        print(log_info)


def val_batch_gcs(resnet, gcsnet, self_attn, classifier, device, loss_func, with_gcs):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((512, 512)),
         transforms.Normalize([0.1697, 0.1697, 0.1697], [0.3019, 0.3019, 0.3019])])
    with torch.no_grad():
        acc = 0.0
        val_loss = 0.0
        count = 0
        val = open("data/val.txt", 'r')
        for item in val:
            count += 1
            item_list = item.split(' ')
            nums = len(os.listdir("data/" + item_list[0]))
            items = os.listdir("data/" + item_list[0])
            temp_acc = 0
            gcs_s = get_gcs(item_list[0])
            imgs = []
            for i in items:
                img = cv2.imread("data/" + item_list[0] + "/" + i)
                imgs.append(numpy.array(transform(img)))
            img_tensor = torch.from_numpy(numpy.array(imgs))
            img = img_tensor.to(device)

            gcs_np = []
            for i in range(15):
                gcs_s -= 1
                if gcs_s >= 0:
                    gcs_np.append(1)
                else:
                    gcs_np.append(0)
            gcs_tensor = torch.FloatTensor(numpy.array([gcs_np] * nums))
            gcs = gcs_tensor.to(device)
            label_tensor = torch.from_numpy(numpy.array([int(item_list[1][0])] * nums)).long()
            label = label_tensor.to(device)

            global_feature = resnet(img)
            if with_gcs:
                gcs_feature = gcsnet(gcs)
                global_feature = torch.cat([global_feature, gcs_feature], dim=1)
                global_feature = torch.unsqueeze(global_feature, 1)
                global_feature = self_attn(global_feature)
            else:
                # global_feature = torch.unsqueeze(global_feature, 1)
                global_feature = global_feature
                # outputs = classify(global_feature)
            outputs = global_feature

            loss = loss_func(outputs, label)
            val_loss += loss.item()
            predictions = torch.max(outputs.data, dim=1)[1]
            temp_acc += torch.eq(predictions, label).sum().item()

            if temp_acc > int(nums / 2):
                acc += 1

        return acc / count, val_loss / count
