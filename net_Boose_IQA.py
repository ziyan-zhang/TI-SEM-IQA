import torch
import torch.nn as nn
import torch.nn.functional as F


class netReg(nn.Module):
    def __init__(self):
        super(netReg, self).__init__()
        self.model_name = 'net_Boose'
        self.conv1_1 = nn.Conv2d(1, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(128, 128, 3)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(128, 256, 3)
        self.conv4_2 = nn.Conv2d(256, 256, 3)
        self.maxpool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(256, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_q1 = nn.Linear(512, 512)
        self.dp = nn.Dropout(0.5)
        self.fc_q2 = nn.Linear(512, 1)
        self.fc_a1 = nn.Linear(512, 512)
        self.fc_a2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x1 = F.relu(self.fc_q1(x))
        x1 = self.dp(x1)
        x1 = self.fc_q2(x1)
        # x1 = self.dp(x1)

        alpha = F.relu(self.fc_a1(x)) + 1e-6
        alpha = self.dp(alpha)
        alpha = self.fc_a2(alpha)
        # alpha = self.dp(alpha)

        return x1, alpha
