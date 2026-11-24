import torch
import json
import matplotlib.pyplot as plt
from nerf.ray import Ray
from nerf.camera import Camera

with open("data/nerf_synthetic/lego/transforms_train.json") as camerasFile:
    camerasDictionary = json.load(camerasFile) # dictionary, a collection of name:value pairs: {"camera_angle_x": 0.6911, "frames": [...]}.

# instanciate a camera for each pose
# In Blender, it was one camera that moved to 100 different places. 
# In my code, a Camera object stores one pose: where the camera stands and which way it faces at one moment, 
# plus its lens' size
cameras = []
camera_angel_x = camerasDictionary["camera_angle_x"]
for frame in camerasDictionary["frames"]:
    cam = Camera(frame["transform_matrix"], 800, 800, camera_angel_x)
    cameras.append(cam)

ax = plt.figure(figsize=(8,8)).add_subplot(projection="3d")
ax.computed_zorder = False

# lego at origin: yellow point
ax.scatter(0,0,0, c="yellow", s=600, zorder=0, label="Lego object")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# cameras' position: blue points
first_camera = True
for cam in cameras:
    p = cam.position
    # ax.scatter(x,y,z, point_size, color)
    ax.scatter(p[0], p[1], p[2], s=10, c="blue", zorder=10, label='cameras' if first_camera else None)
    first_camera = False

# rays: orange lines. only for every 20th cameras. 5 rays (4 corners of img + center)
first_ray = True
for cam in cameras[::20]:
    rays_d, rays_o = cam.rays_gen()
    for (y, x) in [(0,0), (0,799), (799, 0), (799, 799), (399, 399)]:
        d = rays_d[y, x] 
        o = rays_o[y, x]
        end = 2.0 * d + o   # move 2 units forward along the ray
        ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], c="orange", lw=1,
                label="rays (4 img corners + center)" if first_ray else None)
        first_ray = False


# dome: grey wireframe
radius = cameras[0].position.norm()
azimuth = torch.linspace(0, 2 * torch.pi, 40)
elevation = torch.linspace(0, torch.pi / 2, 15)
az, el = torch.meshgrid(azimuth, elevation, indexing="xy")
X = radius * torch.cos(el) * torch.cos(az)
Y = radius * torch.cos(el) * torch.sin(az)
Z = radius * torch.sin(el)
ax.plot_wireframe(X.numpy(), Y.numpy(), Z.numpy(), color="gray", alpha=0.3, lw=0.5, label="cameras dome")

ax.set_aspect("equal")
plt.gcf().canvas.manager.set_window_title("Cameras around Lego")
ax.legend()

plt.show()

