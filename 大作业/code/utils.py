# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/6/18 22:34
"""

import numpy as np
from PIL import Image


def save_image(img, path, nrow=10, padding=5):
    N, C, W, _ = img.shape
    if N % nrow != 0:
        print("N %n row != 0")
        return
    ncol = int(N / nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i * nrow + j])
            img_.append(np.zeros((C, W, padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C, padding, img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C, padding, img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C, img.shape[1] ,padding)), img], 2)
    min_ = img.min()
    max_ = img.max()
    img = (img - min_) / (max_ - min_) * 255
    img = img.transpose((1,2,0))
    if C == 3:
        img = img[:, :, : : -1]
    elif C == 1:
        img = img[:, :, 0]
    Image.fromarray(np.uint8(img)).save(path)
