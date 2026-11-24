import torch
import math

class Camera:
    def __init__(self, cameraToWorld, imgHeight, imgWidth, theta):
        self.cameraToWorld = torch.tensor(cameraToWorld, dtype=torch.float32) #float32 to be accepted by MPS
        self.position = self.cameraToWorld[:3, 3]
        self.rotation = self.cameraToWorld[:3, :3]
        # what the camera sees
        self.height = imgHeight
        self.width = imgWidth
        self.theta = theta
        self.focal = (self.width / 2) / math.tan(self.theta / 2) # f = (W/2) / tan(θ/2)

    def rays_gen(self):
        """generate rays for the all pixels in the image that camera sees"""
        x_axis = torch.arange(self.width, dtype=torch.float32, device=self.position.device) + 0.5 #device==cameradevice so if camera is on MPS the tables are there too, so tensors are on same device
        y_axis = torch.arange(self.height, dtype=torch.float32, device=self.position.device) + 0.5
        x, y = torch.meshgrid(x_axis, y_axis, indexing="xy") #holds the x and y of the center of every pixel

        #direction of each ray in camera space
        rays_direction = torch.stack([
                (x - self.width/2) / self.focal,    # The pixel centers are at x = 0.5, 1.5, 2.5, 3.5 -> (x-w/2): −1.5, −0.5, +0.5, +1.5
                -(y - self.height/2) / self.focal,  # - : top lines positive, bottom line negative
                -torch.ones_like(x)     # camera looks along -z, so one step forward is -1 
        ], dim = -1) # dim=-1 puts each x,y,z together 
        #dim= 1  →  each line keeps the tables apart:   [x,x,x]  [y,y,y]  [z,z,z]
        #dim=-1 →  each pixel gets its own triple:     [x,y,z]  [x,y,z]  [x,y,z]
        #

        # rays in world space
        rays_direction_world = rays_direction @ self.rotation.T # d.T * R.T = (R * d).T and d is already transposed

        rays_origin = self.position.expand(self.height, self.width, 3) # for all pixels store camera position which has 3 elements

        return rays_direction_world, rays_origin
    