import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchsummary


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
        x = x.reshape(x.shape[0], -1)
        x = self.final(x)
        return x


# BasicBlock for ResNet18
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, padding=None, change_channels=False) -> None:
        super(BasicBlock, self).__init__()
        if padding is None:
            padding = [1, 1]
        self.basicBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=strides[0], padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=strides[1], padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        # for each ConvBlock(build by ResBlocks), first ResBlock should change source(x) channels to shortcut
        if change_channels and strides[0] != [1, 1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=strides[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.basicBlock(x)
        out += self.shortcut(x)
        out = func.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=2) -> None:
        super(ResNet18, self).__init__()
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        # max_pooling
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        self.temp_channels = 64

        self.conv2 = self._generate_network_layers(BasicBlock, 64, strides=[[1, 1], [1, 1]])
        self.conv3 = self._generate_network_layers(BasicBlock, 128, strides=[[2, 1], [1, 1]])
        self.conv4 = self._generate_network_layers(BasicBlock, 256, strides=[[2, 1], [1, 1]])
        self.conv5 = self._generate_network_layers(BasicBlock, 512, strides=[[2, 1], [1, 1]])
        # average_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(512, 2)

    def _generate_network_layers(self, BasicBlock, out_channels, strides):
        layers = [BasicBlock(self.temp_channels, out_channels, strides[0], change_channels=True)]
        # first BasicBlock should change channels
        self.temp_channels = out_channels
        layers.append(BasicBlock(self.temp_channels, out_channels, strides[1], change_channels=False))
        self.temp_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final(x)
        return x


class IMG_net(nn.Module):
    def __init__(self):
        """
        the basic backbone of ResNet-50
        """
        super(IMG_net, self).__init__()
        BackBone = torchvision.models.__dict__['resnet50'](pretrained=True)

        add_block = []
        add_block += [nn.Linear(1000, 512)]
        add_block += [nn.ReLU(True)]
        add_block += [nn.Dropout(0.15)]
        add_block += [nn.Linear(512, 128)]

        add_block = nn.Sequential(*add_block)
        self.BackBone = BackBone
        self.add_block = add_block

    def forward(self, x):
        x = self.BackBone(x)
        x = self.add_block(x)
        return x


class GCS_net(nn.Module):
    def __init__(self):
        """
        MLP, use to process gcs score
        """
        super(GCS_net, self).__init__()
        self.trans1 = nn.Sequential(
            nn.Linear(15, 32)
            , nn.ReLU(inplace=True)
            , nn.Linear(32, 128)
            , nn.ReLU(inplace=True)
            , nn.Linear(128, 256)
            , nn.ReLU(inplace=True)
            , nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.trans1(x)
        return x


# abandoned
# class head(nn.Module):
#     def __init__(self):
#         super(head, self).__init__()
#         self.classfy1 = nn.Sequential(
#             nn.Linear(192, 512)
#             , nn.ReLU(inplace=True)
#             , nn.Linear(512, 128)
#             , nn.ReLU(inplace=True)
#             , nn.Linear(128, 16)
#             , nn.ReLU(inplace=True)
#             , nn.Linear(16, 2)
#         )
#
#     def forward(self, x):
#         x = self.classfy1(x)
#         return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        """
        SelfAttention Module, use to transform global feature created by image and gcs score
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(torch.nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)


class Transition(torch.nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm1d(channels))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool1d(2))


class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=1, middele_channels=128,
                 classes=5):
        """
        1D-DenseNet Module, use to conv global feature and generate final target
        """
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 2, 1)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, classes),

        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)

        return x


# run to generate global summary
if __name__ == '__main__':
    input = torch.randn(size=(1, 1, 224))
    denseNet = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=1, classes=2)
    output = denseNet(input)
    print(output.shape)
    torchsummary.summary(GCS_net(), (1, 15), device='cpu')
    torchsummary.summary(ResNet50(ResBlock), (3, 512, 512), device='cpu')
    torchsummary.summary(SelfAttention(16, 192, 192, 0), (1, 192), device='cpu')
    torchsummary.summary(denseNet, (1, 192), device='cpu')

