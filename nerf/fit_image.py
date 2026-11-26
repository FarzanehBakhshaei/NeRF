import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio
from ImageMLP import ImageMLP

class OutputImage:
    """The image the network has to learn. Input:(x,y) -> Output:(r,g,b)"""

    def __init__(self):
        imgRGBAs = iio.imread("data/nerf_synthetic/lego/train/r_4.png") # ndarray. shape in this example: (800, 800, 4)
        img = torch.tensor(imgRGBAs, dtype=torch.float32) / 255.0  # Note:imshow expect values between 0 and 1 and also in alpha blending alpha should be in [0, 1]

        rgb = img[:, :, :3] # (800, 800, [0th,1st,2nd channels])
        alpha = img[:, :, 3:]  # (800, 800, 3rd channel)
        background = torch.tensor([1.0, 1.0, 1.0])
        self.rgb = rgb * alpha + background * (1-alpha)
        self.height = self.rgb.shape[0] # here 800 in shape of rgb of lego (800, 800, 3)
        self.width = self.rgb.shape[1]

    def training_data(self):
        x_axis = torch.arange(self.width, dtype=torch.float32) / (self.width - 1) # 0 to 1
        y_axis = torch.arange(self.height, dtype=torch.float32) / (self.height - 1)
        x, y = torch.meshgrid(x_axis, y_axis, indexing = "xy")
        coordinate = torch.stack([x,y], dim=-1).reshape(-1, 2) # for (800, 800, 2) -> turns to (640000, 2)

        colors = self.rgb.reshape(-1, 3) # self.rgb.shape (H, W, 3) = (800, 800, 3) -> (640000, 3)

        return coordinate, colors


output = OutputImage() # 1. create the image
coords, colors = output.training_data() # 2. get its coords and colors
# choose the device
device = "mps"
model = ImageMLP().to(device)
coords, colors = coords.to(device), colors.to(device)
# choosed adam as optimizer with learning rate 5e-4 like the paper to find the loss' minima
# adam adapt the step size for each parameter
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# fiting the model. training the model with one image now
for step in range(5000):
    idx = torch.randint(0, coords.shape[0], (4096,), device=device) # creates 2^12 random integers from 0 to 800*800 on MPS
    predicted_colors = model(coords[idx])
    loss = ((predicted_colors - colors[idx]) ** 2).mean() # MSE loss. (sigma ||y-ypred||^2) * 1/3N
    loss.backward() # fills w.grad and b,grad
    optimizer.step() # updates w and b.
    optimizer.zero_grad() # clears w.grad and b.grad
    if step % 500 == 0:
        print(step, loss.item())

#print(coords.shape, colors.shape)
#plt.imshow(output.rgb)
with torch.no_grad(): # so we dont train anymore
    learned = model(coords).reshape(output.height, output.width, 3).cpu()
#plt.imshow(learned)
fig, (left, right) = plt.subplots(1, 2) # 1 row, 2 columns (one for each picture)
left.imshow(output.rgb)
left.set_title("Original")
right.imshow(learned)
right.set_title("Learned without positional encoding")
plt.show()