from net.ResNet50 import *

if __name__ == "__main__":
    net = ResNet50(ResBlock)
    print(net)
