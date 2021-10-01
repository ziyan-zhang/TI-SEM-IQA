"""-----------------------------------------------------
创建时间 :  2020/6/11  22:27
说明    :
todo   :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from pathlib import Path


class netReg(nn.Module):  # 3085185个参数
    def __init__(self):
        super(netReg, self).__init__()
        self.model_name = Path(__file__).name[:-3]
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),  # 64, 128  0
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),  # 64, 128  0
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # 64, 128  0
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 64, 128  0
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # 64, 128  0
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # 64, 128  0
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # 64, 128  0
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),  # 64, 128  0
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),  # 64, 128  0
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),  # 64, 128  0

        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 2048)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 1)
        self.dp2 = nn.Dropout(0.5)


    def forward(self, input):
        output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.linear1(output)
        output = self.dp1(output)
        output = self.linear2(output)
        output = self.dp2(output)

        return output

