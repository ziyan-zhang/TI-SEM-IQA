import torch
import torchvision.models as models
import torch.nn as nn


netReg = models.alexnet(pretrained=True)
# netReg = models.alexnet(pretrained=False)

netReg.classifier._modules['6'] = nn.Linear(4096, 1)

# a = torch.randn(10, 3, 256, 256)
# b = netReg(a)