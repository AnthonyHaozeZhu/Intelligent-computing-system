# -*- coding: UTF-8 -*-
"""
@Project ：GAN 
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/6/22 20:57
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.de_conv1 = nn.ConvTranspose2d(100, d*8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        self.de_conv1_bn = nn.BatchNorm2d(d*8)
        self.de_conv2 = nn.ConvTranspose2d(d*8, d*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv2_bn = nn.BatchNorm2d(d*4)
        self.de_conv3 = nn.ConvTranspose2d(d*4, d*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv3_bn = nn.BatchNorm2d(d*2)
        self.de_conv4 = nn.ConvTranspose2d(d*2, d, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.de_conv4_bn = nn.BatchNorm2d(d)
        self.de_conv5 = nn.ConvTranspose2d(d, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, in_put):
        x = in_put.reshape((-1, 100, 1, 1))
        x = F.relu(self.de_conv1_bn(self.de_conv1(x)))
        x = F.relu(self.de_conv2_bn(self.de_conv2(x)))
        x = F.relu(self.de_conv3_bn(self.de_conv3(x)))
        x = F.relu(self.de_conv4_bn(self.de_conv4(x)))
        x = torch.tanh(self.de_conv5(x))
        
        return x


class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(d, d*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))

    def weight_init(self, mean, std):
        for i in self._modules:
            normal_init(self._modules[i], mean, std)

    def forward(self, in_put):
        x = F.leaky_relu(self.conv1(in_put), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


