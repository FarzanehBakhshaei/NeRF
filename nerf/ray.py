import torch
import math

class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin

    def end_of_ray_at(self, t):
        return self.direction * t + self.origin

    