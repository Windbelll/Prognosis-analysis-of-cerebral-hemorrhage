import cv2
import matplotlib.pyplot as plt
import torch
from torchcam.methods import SmoothGradCAMpp, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

from net.net_archs import Test

# model = net.ResNet50(net.ResBlock)
# model = torch.load("../best_batch.pth")
model = Test(gcs=9, with_gcs=False)
cam_extractor = GradCAM(model)
test_image = cv2.imread("../data/bad/cao_wanfa_9/12.png")
transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((512, 512)),
     transforms.Normalize([0.1747, 0.1747, 0.1747], [0.3030, 0.3030, 0.3030])])
tensor = transformer(test_image)
tensor = torch.unsqueeze(tensor, dim=0)
tensor = tensor.cuda()

out = model(tensor)
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(test_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.show()
