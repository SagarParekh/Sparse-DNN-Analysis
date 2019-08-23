#!/usr/bin/env python3
import os
from torchvision import models
import torch
import torch.nn as nn

# #Shows how to extract features
# l=nn.Linear(3,5)
# w=list(l.parameters())
# print(w)
#
# for p in l.parameters():
#     print(p.name,p.data)

alexnet = models.alexnet(pretrained=True)
# print(alexnet)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image
img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

alexnet.eval()

out=alexnet(batch_t)
# print("\n\n",out.shape)

with open('imagenet1000_clsidx_to_labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]
# print(classes[0])

_, index = torch.max(out, 1)
# print(index[0])
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(percentage[208])
print(classes[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])

# Do pip install torchsummary
from torchsummary import summary
summary(alexnet,(3,224,224))
