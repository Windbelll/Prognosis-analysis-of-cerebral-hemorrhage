import argparse

import torch.utils.data
import torchvision
from torchvision.transforms import transforms

from net.net_archs import *
from net.net_apis import run, val_without_info


def prepare_to_train(new_model, load_dict_path, dataset_dir, batch_size, epochs, learning_rate, use_gpu,
                     use_transforms):
    net = ResNet50(ResBlock)
    # net = ResNet18(BasicBlock)
    if not new_model:
        net.load_state_dict(torch.load(load_dict_path))
        print("using local dict: " + load_dict_path)
    transform = transforms.Compose([transforms.ToTensor()])
    if use_transforms:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    print("Basic model & normalization module loaded")

    train_data = torchvision.datasets.ImageFolder(dataset_dir + "/train", transform=transform)
    val_data = torchvision.datasets.ImageFolder(dataset_dir + "/val", transform=transform)

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
    last = run(net=net, device=device, epochs=epochs, train_loader=train_loader, val_loader=val_loader,
               optimizer=optimizer, loss_func=loss_func, train_c=len(train_data), val_c=len(val_data))
    torch.save(last, "./logs/last.pth")


def test_val(new_model, load_dict_path, dataset_dir, batch_size, epochs, learning_rate, use_gpu,
             use_transforms):
    # net = ResNet50(ResBlock)
    # net = ResNet18(BasicBlock)
    net = torch.load(load_dict_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    if use_transforms:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_data = torchvision.datasets.ImageFolder(dataset_dir + "/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    loss_func = nn.CrossEntropyLoss()
    test_accuracy, test_loss = val_without_info(net, test_loader, device, loss_func=loss_func, val_c=len(test_data))
    print(test_accuracy)
    print(test_loss)


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to training")
    parser.add_argument("--new_model", default=True, help="program will create a new model if True", type=bool)
    parser.add_argument("--load_dict_path", default="D:/A-exps-logs/src_res50_epoch300_lr0.0001/best_every.pth",
                        help="the path of old model")
    parser.add_argument("--dataset_dir", default="./dataset", help="the path of dataset which should be (here)/train..")
    parser.add_argument("--batch_size", default=16, help="the batch_size of training")
    parser.add_argument("--epochs", default=100, help="the epochs of training", type=int)
    parser.add_argument("--use_gpu", default=True, help="device choice, if cuda isn't available program will warn",
                        type=bool)
    parser.add_argument("--learning_rate", default=0.00001, help="learning_rate", type=float)
    parser.add_argument("--use_transforms", default=True, help="using transform to normalize data", type=bool)
    parser.add_argument("--debug", default=False, help="debug", type=bool)

    args = parser.parse_args()
    print(args)
    if not args.debug:
        prepare_to_train(new_model=args.new_model, load_dict_path=args.load_dict_path, dataset_dir=args.dataset_dir,
                         batch_size=int(args.batch_size), epochs=args.epochs, learning_rate=args.learning_rate,
                         use_gpu=args.use_gpu, use_transforms=args.use_transforms)
    else:
        test_val(new_model=args.new_model, load_dict_path=args.load_dict_path, dataset_dir=args.dataset_dir,
                 batch_size=int(args.batch_size), epochs=args.epochs, learning_rate=args.learning_rate,
                 use_gpu=args.use_gpu, use_transforms=args.use_transforms)
