#!/usr/bin/env python3
from torchvision import models
import torch

# print(dir(models))

alexnet = models.alexnet(pretrained=True)
print(alexnet)
