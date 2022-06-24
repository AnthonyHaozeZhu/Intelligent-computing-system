# # -*- coding: UTF-8 -*-
# """
# @Project ：code
# @File ：model.py
# @Author ：AnthonyZ
# @Date ：2022/6/18 20:17
# """
#
# import torch
# import torch.nn as nn
#
#
# # class Generator(nn.Module):
# #     def __init__(self, opt):
# #         super(Generator, self).__init__()
# #         self.args = opt
# #         self.linear1 = nn.Linear(100, 4 * 4 * 16)
# #         self.conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
# #         self.relu1 = nn.ReLU()
# #         self.bn1 = nn.BatchNorm2d(64)
# #         self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
# #         self.relu2 = nn.ReLU()
# #         self.bn2 = nn.BatchNorm2d(64)
# #         self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=(2, 2), kernel_size=(5, 5))
# #         self.relu3 = nn.ReLU()
# #         self.bn3 = nn.BatchNorm2d(64)
# #         self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=(1, 1), kernel_size=(5, 5))
# #         self.linear2 = nn.Linear(3*57*153, self.args.channels * self.args.img_w * self.args.img_h)
# #         self.Tanh = nn.Tanh()
# #
# #     def forward(self, x):
# #         x = self.linear1(x)
# #         x = x.view((-1, 4, 4, 16))
# #         x = self.bn1(self.relu1(self.conv1(x)))
# #         x = self.bn2(self.relu2(self.conv2(x)))
# #         x = self.bn3(self.relu3(self.conv3(x)))
# #         x = self.Tanh(self.conv4(x))
# #         x = x.view((-1, 3*57*153))
# #         x = self.linear2(x)
# #         x = x.view((-1, self.args.channels, self.args.img_w, self.args.img_h))
# #         return x
# #
# #
# # class Discriminator(nn.Module):
# #     def __init__(self, opt):
# #         super(Discriminator, self).__init__()
# #         self.args = opt
# #         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
# #         self.relu1 = nn.ReLU()
# #         self.bn1 = nn.BatchNorm2d(64)
# #         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
# #         self.relu2 = nn.ReLU()
# #         self.bn2 = nn.BatchNorm2d(64)
# #         self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=(2, 2))
# #         self.relu3 = nn.ReLU()
# #         self.bn3 = nn.BatchNorm2d(3)
# #         self.linear1 = nn.Linear(3*29*13, 256)
# #         self.Tanh = nn.Tanh()
# #         self.linear2 = nn.Linear(256, 1)
# #
# #     def forward(self, x):
# #         x = self.bn1(self.relu1(self.conv1(x)))
# #         x = self.bn2(self.relu2(self.conv2(x)))
# #         x = self.bn3(self.relu3(self.conv3(x)))
# #         x = x.view(-1, 3*29*13)
# #         x = self.linear2(self.Tanh(self.linear1(x)))
# #         return torch.sigmoid(x)
#
#
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()
# #
# #         def discriminator_block(in_filters, out_filters, bn=True):
# #             block = [nn.Conv2d(in_filters, out_filters, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
# #             if bn:
# #                 block.append(nn.BatchNorm2d(out_filters, 0.8))
# #             return block
# #
# #         self.model = nn.Sequential(
# #             *discriminator_block(3, 16, bn=False),
# #             *discriminator_block(16, 32),
# #             *discriminator_block(32, 64),
# #             *discriminator_block(64, 128),
# #             *discriminator_block(128, 128),
# #         )
# #
# #         # The height and width of downsampled image
# #         # ds_size = opt.img_size // 2 ** 4
# #         self.adv_layer = nn.Sequential(nn.Linear(32 * 64, 1), nn.Sigmoid())
# #
# #     def forward(self, img):
# #         out = self.model(img)
# #         out = out.view(out.shape[0], -1)
# #         validity = self.adv_layer(out)
# #         return validity
# #
# #
# # class Generator(nn.Module):
# #     def __init__(self):
# #         super(Generator, self).__init__()
# #
# #         # self.init_size = opt.img_size // 4
# #         self.l1 = nn.Sequential(nn.Linear(100, 4 * 4 * 16))
# #
# #         self.conv_blocks = nn.Sequential(
# #             nn.ConvTranspose2d(16, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # 4*4 => 8*8
# #             nn.BatchNorm2d(128),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # nn.Upsample(scale_factor=2),
# #             nn.ConvTranspose2d(128, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),  # 4*4 => 8*8
# #             nn.BatchNorm2d(128, 0.8),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # nn.Upsample(scale_factor=2),
# #             nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),  # 16*16 => 32*32
# #             nn.BatchNorm2d(64, 0.8),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # nn.Upsample(scale_factor=2),
# #             nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),  # 32*32 => 64*64
# #             nn.BatchNorm2d(64, 0.8),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # nn.Upsample(scale_factor=2),
# #             nn.ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # 64*64 => 128*128
# #             nn.Tanh(),
# #         )
# #
# #     def forward(self, z):
# #         out = self.l1(z)
# #         out = out.view(out.shape[0], 16, 4, 4)
# #         img = self.conv_blocks(out)
# #         return img
#
#
# class Discriminator(nn.Module) :
#     def __init__(self, input_channel, ndf=64):
#         super(Discriminator, self).__init__()
#         # 256 * 256
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         # 128 * 128
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         # 64 * 64
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         # 32 * 32
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         # 16 * 16
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(ndf * 8, ndf, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
#         # 8 * 8
#         self.fulc = nn.Sequential(
#             nn.Linear(8 * 8 * ndf, ndf),
#             nn.ReLU(),
#             nn.Linear(ndf, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         layer1_out = self.layer1(x)
#         layer2_out = self.layer2(layer1_out)
#         layer3_out = self.layer3(layer2_out)
#         layer4_out = self.layer4(layer3_out)
#         layer5_out = self.layer5(layer4_out)
#         feat = torch.flatten(layer5_out, start_dim=1)
#         cls = self.fulc(feat)
#
#         return cls
#
#
# class Generator(nn.Module):
#     def __init__(self, input_channel=3, output_channel=3, ngf=64):
#         super(Generator, self).__init__()
#         # 100
#         self.linear = nn.Linear(100, ngf * 8 * 8)
#         # 64 * 8 * 8
#
#         # 512 * 16 * 16
#         self.de1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(ngf, ngf * 8, kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(ngf * 8),
#             nn.Dropout(p=0.5)
#         )
#         # 256 * 32 * 32
#         self.de2 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(ngf * 8),
#             nn.Dropout(p=0.5)
#         )
#         # 128 * 64 * 64
#         self.de3 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(ngf * 8),
#             nn.Dropout(p=0.5)
#         )
#         # 64 * 128 * 128
#         self.de4 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(ngf * 8),
#             nn.Dropout(p=0.5)
#         )
#         # 3 * 256 * 256
#         self.de5 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(ngf, output_channel, kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(ngf * 4),
#         )
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = x.reshape(-1, 64, 8, 8)
#         # Encoder
#         en1_out = self.de1(x)
#         en2_out = self.de2(en1_out)
#         en3_out = self.de3(en2_out)
#         en4_out = self.de4(en3_out)
#         en5_out = self.de5(en4_out)
#
#         return en5_out
#
#
#


# -*- coding: UTF-8 -*-
"""
@Project ：code
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/6/18 20:17
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module) :
    def __init__(self, input_channel, ndf=64):
        super(Discriminator, self).__init__()
        # 256 * 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 16 * 16
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # 8 * 8
        self.fulc = nn.Sequential(
            nn.Linear(8 * 8 * ndf, ndf),
            nn.ReLU(),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        feat = torch.flatten(layer5_out, start_dim=1)
        cls = self.fulc(feat)

        return cls


class Generator(nn.Module):
    def __init__(self, input_channel=3, output_channel=3, ngf=64):
        super(Generator, self).__init__()
        # 100
        self.linear = nn.Linear(100, ngf * 8 * 8)
        # 64 * 8 * 8

        # 512 * 16 * 16
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf * 8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 256 * 32 * 32
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 128 * 64 * 64
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 64 * 128 * 128
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 3 * 256 * 256
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, output_channel, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(ngf * 4),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, 64, 8, 8)
        # Encoder
        en1_out = self.de1(x)
        en2_out = self.de2(en1_out)
        en3_out = self.de3(en2_out)
        en4_out = self.de4(en3_out)
        en5_out = self.de5(en4_out)

        return en5_out