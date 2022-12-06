import argparse

import torch.utils.data
import torchvision
from torchvision.transforms import transforms

from net.ResNet50 import *
from net.train import train


def prepare_to_train(new_model, load_dict_path, dataset_dir, batch_size, epochs, learning_rate, use_gpu, use_transforms):
    net = ResNet50(ResBlock)
    if not new_model:
        net.load_state_dict(torch.load(load_dict_path))
        print("using local dict: " + load_dict_path)
    transform = transforms.Compose([])
    if use_transforms:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    print("Basic model & normalization module loaded")

    train_data = torchvision.datasets.ImageFolder(dataset_dir + "/train", transform=transform)
    val_data = torchvision.datasets.ImageFolder(dataset_dir + "/val", transform=transform)
    test_data = torchvision.datasets.ImageFolder(dataset_dir + "/test", transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size)
    print("datasets loaded, training_sum: %d, valid_sum: %d" % (len(train_data), len(val_data)))

    loss_func = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("use device {}".format(device))
    print("prepare completed! launch training!\U0001F680")
    print("test out")
    exit(0)
    train(net=net, device=device, epochs=epochs, train_loader=train_loader, val_loader=val_loader,
          optimizer=optimizer, loss_func=loss_func)


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to training")
    parser.add_argument("--new_model", default=True, help="program will create a new model if True")
    parser.add_argument("--load_dict_path", default="./logs/best.pth", help="the path of old model")
    parser.add_argument("--dataset_dir", default="./dataset", help="the path of dataset which should be (here)/train..")
    parser.add_argument("--batch_size", default=4, help="the batch_size of training")
    parser.add_argument("--epochs", default=200, help="the epochs of training")
    parser.add_argument("--use_gpu", default=True, help="device choice, if cuda isn't available program will warn")
    parser.add_argument("--learning_rate", default=0.001, help="learning_rate")
    parser.add_argument("--use_transforms", default=True, help="using transform to normalize data")

    args = parser.parse_args()
    print(args)
    prepare_to_train(new_model=args.new_model, load_dict_path=args.load_dict_path, dataset_dir=args.dataset_dir,
                     batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
                     use_gpu=args.use_gpu, use_transforms=args.use_transforms)
