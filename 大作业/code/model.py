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
        self.linear1 = nn.Linear(100, 128)
        self.linear2 = nn.Linear(128, 256)
        self.batch_normal1 = nn.BatchNorm1d(256, 0.8)
        self.linear3 = nn.Linear(256, 512)
        self.batch_normal2 = nn.BatchNorm1d(512, 0.8)
        self.linear4 = nn.Linear(512, 1024)
        self.batch_normal3 = nn.BatchNorm1d(1024, 0.8)
        self.linear5 = nn.Linear(1024, int(opt.channels * opt.img_w * opt.img_h))
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.linear1(noise)
        x = self.linear2(self.relu(x))
        x = self.linear3(self.relu(self.batch_normal1(x)))
        x = self.linear4(self.relu(self.batch_normal2(x)))
        x = self.linear5(self.relu(self.batch_normal3(x)))
        x = self.tanh(x)
        x = x.view((-1, self.args.channels, self.args.img_w, self.args.img_h))
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.args = opt
        self.linear1 = nn.Linear((opt.channels * opt.img_w * opt.img_h), 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)
        self.drop_out = nn.Dropout(0.4)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, img):
        img = img.view((img.shape[0], (-1)))
        x = self.linear1(img)
        x = self.linear2(self.relu(x))
        x = self.linear3(self.drop_out(self.relu(x)))
        x = self.linear4(self.drop_out(self.relu(x)))
        return x

