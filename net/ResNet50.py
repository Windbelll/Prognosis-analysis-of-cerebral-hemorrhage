import torch
import torch.nn as nn
import torch.nn.functional as func


# Bottleneck for ResNet50, conv1, 3 use to transform channels, k=(1, 1)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, paddings, change_channels=False) -> None:
        super(ResBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=strides[0], padding=paddings[0],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # replace x
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=strides[1], padding=paddings[1],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=(1, 1), stride=strides[2], padding=paddings[2],
                      bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        # for each ConvBlock(build by ResBlocks), first ResBlock should change source(x) channels to shortcut
        if change_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=(1, 1), stride=strides[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = func.relu(out)
        return out


# use to classify GOS score (1~5)
class ResNet50(nn.Module):
    def __init__(self, ResBlock, num_classes=2) -> None:
        super(ResNet50, self).__init__()
        # conv1 (224,224,3) -> (112,112,64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        # max_pooling (112,112,64) -> (56,56,64)
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        self.temp_channels = 64
        # conv2 (56,56,64) -> (56,56,256) needn't down channels in first ResBlock
        self.conv2 = self._generate_network_layers(ResBlock, 64, strides=[[1, 1, 1]] * 3)
        # conv3 (56,56,128) -> (28,28,512)
        self.conv3 = self._generate_network_layers(ResBlock, 128, strides=[[1, 2, 1]] + [[1, 1, 1]] * 3)
        # conv4 (28,28,256) -> (14,14,1024)
        self.conv4 = self._generate_network_layers(ResBlock, 256, strides=[[1, 2, 1]] + [[1, 1, 1]] * 5)
        # conv5 (14,14,512) -> (7,7,2048)
        self.conv5 = self._generate_network_layers(ResBlock, 512, strides=[[1, 2, 1]] + [[1, 1, 1]] * 2)
        # average_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(2048, num_classes)

    def _generate_network_layers(self, ResBlock, first_out_channels, strides):
        layer_num = len(strides)
        paddings = [[0, 1, 0]] * layer_num
        # first ResBlock should change channels
        layers = [ResBlock(self.temp_channels, first_out_channels, strides[0], paddings[0], change_channels=True)]
        # for temp_channels, ResNet50 always * 4 between every ResBlock
        self.temp_channels = 4 * first_out_channels
        for i in range(1, layer_num):
            layers.append(
                ResBlock(self.temp_channels, first_out_channels, strides[i], paddings[i], change_channels=False))
            self.temp_channels = 4 * first_out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x

