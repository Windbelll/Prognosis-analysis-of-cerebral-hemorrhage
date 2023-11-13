import argparse

import torch.utils.data

from data.dataset import gcs_dataset
from net.net_apis import run_gcs
from net.net_archs import *


def prepare_to_train(batch_size, epochs, learning_rate, use_gpu, with_gcs):
    if with_gcs:
        img_net = IMG_net()
        gcs_net = GCS_net()
        classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)
        self_attn = SelfAttention(16, 192, 192, 0)
        params = [p for p in img_net.parameters() if p.requires_grad]
        print("step1: " + str(len(params)))
        params += [p for p in gcs_net.parameters() if p.requires_grad]
        print("step2: " + str(len(params)))
        params += [p for p in classify_head.parameters() if p.requires_grad]
        print("step3: " + str(len(params)))
        params += [p for p in self_attn.parameters() if p.requires_grad]
        print("step4: " + str(len(params)))
    else:
        img_net = IMG_net()
        classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)
        gcs_net = None
        self_attn = None
        params = [p for p in img_net.parameters() if p.requires_grad]
        print("step1: " + str(len(params)))
        # params += [p for p in classify_head.parameters() if p.requires_grad]
        # print("step3: " + str(len(params)))
    # print(img_net)
    train_set = gcs_dataset("data/train.txt")
    val_set = gcs_dataset("data/val.txt")
    val_set += gcs_dataset("data/test.txt")

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size)
    print("datasets loaded, training_sum: %d, valid_sum: %d" % (len(train_set), len(val_set)))

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("use device {}".format(device))
    print("prepare completed! launch training!\U0001F680")
    run_gcs(resnet=img_net, gcsnet=gcs_net, classify=classify_head, device=device, epochs=epochs,
            train_loader=train_loader, val_loader=val_loader, self_attn=self_attn, with_gcs=with_gcs,
            optimizer=optimizer, loss_func=loss_func, train_c=len(train_set), val_c=len(val_set))


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to training")
    parser.add_argument("--batch_size", default=4, help="the batch_size of training")
    parser.add_argument("--epochs", default=300, help="the epochs of training", type=int)
    parser.add_argument("--use_gpu", default=True, help="device choice, if cuda isn't available program will warn",
                        type=bool)
    parser.add_argument("--learning_rate", default=0.0001, help="learning_rate", type=float)
    parser.add_argument("--with_gcs", default=True, help="using gcs feature", type=bool)

    args = parser.parse_args()
    print(args)

    prepare_to_train(batch_size=int(args.batch_size), epochs=args.epochs, learning_rate=args.learning_rate,
                     use_gpu=args.use_gpu,  with_gcs=args.with_gcs)
