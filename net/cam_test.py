import matplotlib.pyplot as plt
import numpy as np

import numpy
import torch
import cv2
from torchvision.transforms import transforms
import cam as c
import net.net_archs as net

# model = net.ResNet50(net.ResBlock)
model = torch.load("../best_batch.pth")


test_image = cv2.imread("../data/bad/cai_mingli_12/12.png")
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
tensor = transformer(test_image)
tensor = torch.unsqueeze(tensor, dim=0)

label_tensor = torch.from_numpy(numpy.array([int(0)])).long()
label_tensor = label_tensor.cuda()
target_layers = [model.conv5]

# src = cv2.resize(test_image, (224, 224))
cam = c.GradCAM(model=model, target_layers=target_layers, use_cuda=True)
gray = cam(tensor, label_tensor)
gray = gray[0]
dst = c.show_cam_on_image(test_image.astype(dtype=np.float32) / 255.,
                          gray, True)
cv2.imshow("cam", dst)
cv2.waitKey(0)
