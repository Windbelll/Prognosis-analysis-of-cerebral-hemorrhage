import argparse

import cv2
import matplotlib.pyplot as plt
import torch
from torchcam.methods import SmoothGradCAMpp, GradCAM, XGradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

from net.net_archs import Test

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to training")
    parser.add_argument("--path", default="example/sample3.png", help="the batch_size of training")
    parser.add_argument("--GCS", default=14, type=int, help="the epochs of training")
    parser.add_argument("--with_gcs", default=False, type=bool, help="using gcs feature")

    args = parser.parse_args()

    model1 = Test(gcs=args.GCS, select=10, with_gcs=args.with_gcs)
    cam_extractor = SmoothGradCAMpp(model1)

    test_image = cv2.imread(args.path)

    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((512, 512)),
         transforms.Normalize([0.1697, 0.1697, 0.1697], [0.3019, 0.3019, 0.3019])])
    tensor = transformer(test_image)
    tensor = torch.unsqueeze(tensor, dim=0)
    tensor = tensor.cuda()

    out = model1(tensor)
    print(out)

    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    result = overlay_mask(to_pil_image(test_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    if args.with_gcs:
        plt.savefig("GCS" + args.path)
    else:
        plt.savefig("noGCS" + args.path)
    plt.show()

