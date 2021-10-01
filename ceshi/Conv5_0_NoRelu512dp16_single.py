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
            nn.Conv2d(1,64,4,2,1, bias=False),  # 64, 128  0
            nn.LeakyReLU(0.2, inplace=True),  # 1

            nn.Conv2d(64,64,4,2,1, bias=False),  # 64, 64  2
            nn.BatchNorm2d(64),  # 3
            nn.LeakyReLU(0.2, inplace=True),  # 4

            nn.Conv2d(64,64*2,4,2,1, bias=False),  # 128, 64  5
            nn.BatchNorm2d(64*2),  # 6
            nn.LeakyReLU(0.2, inplace=True),  # 7

            nn.Conv2d(64*2,64*4, 4, 2, 1, bias=False),  # 256, 16  # 8
            nn.BatchNorm2d(64*4),  # 9
            nn.LeakyReLU(0.2, inplace=True),  # 10

            nn.Conv2d(64*4,64*8, 4, 2, 1, bias=False),  # 512, 8  11
            nn.BatchNorm2d(64*8),  # 12
            nn.LeakyReLU(0.2, inplace=True),  #13
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 16)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(16, 1)


    def forward(self, input):
        output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.linear1(output)
        output = self.dp1(output)
        output = self.linear2(output)

        return output
