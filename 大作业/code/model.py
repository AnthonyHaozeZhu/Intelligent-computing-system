# -*- coding: UTF-8 -*-
"""
@Project ：code
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/6/18 20:17
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.args = opt
        self.linear1 = nn.Linear(100, 4 * 4 * 16)
        self.conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=(1, 1), kernel_size=(5, 5))
        self.linear2 = nn.Linear(3*57*153, self.args.channels * self.args.img_w * self.args.img_h)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = x.view((-1, 4, 4, 16))
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.Tanh(self.conv4(x))
        x = x.view((-1, 3*57*153))
        x = self.linear2(x)
        x = x.view((-1, self.args.channels, self.args.img_w, self.args.img_h))
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.args = opt
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=(2, 2))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(3)
        self.linear1 = nn.Linear(3*29*13, 256)
        self.Tanh = nn.Tanh()
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = x.view(-1, 3*29*13)
        x = self.linear2(self.Tanh(self.linear1(x)))
        return torch.sigmoid(x)


