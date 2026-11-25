import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio

class OutputImage:
    """The image the network has to learn. Input:(x,y) -> Output:(r,g,b)"""

    def __init__(self):
        imgRGBAs = iio.imread("data/nerf_synthetic/lego/train/r_4.png") # ndarray. shape in this example: (800, 800, 4)
        img = torch.tensor(imgRGBAs, dtype=torch.float32) / 255.0  # Note:imshow expect values between 0 and 1 and also in alpha blending alpha should be in [0, 1]

        rgb = img[:, :, :3] # (800, 800, [0th,1st,2nd channels])
        alpha = img[:, :, 3:]  # (800, 800, 3rd channel)
        background = torch.tensor([1, 1, 1])
        self.rgb = rgb * alpha + background * (1-alpha)
        self.height = self.rgb.shape[0] # here 800 in shape of rgb of lego (800, 800, 3)
        self.width = self.rgb.shape[1]


output = OutputImage()
plt.imshow(output.rgb)
plt.show()