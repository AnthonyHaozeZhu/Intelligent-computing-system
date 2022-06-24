# -*- coding: UTF-8 -*-
"""
@Project ：GAN 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/6/22 21:09
"""

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from utils import *

img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # transform to [0, 1]
    # transform to [-1, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImageDataset(Dataset):
    def __init__(self, path, vec_size=(1, 256, 256), transform=None):
        self.image_paths = get_all_image_path(path)
        self.transform = transform
        self.random_vector = [torch.normal(0, 1, vec_size) for _ in range(len(self.image_paths))]
        assert len(self.random_vector) == len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        image = self.transform(image)
        return image, self.random_vector[item]

    def __len__(self):
        return len(self.image_paths)


def get_dataloader(opt):
    dataset = ImageDataset(path=opt.dataset, vec_size=(1, 256, 256), transform=img_transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=int(opt.batch_size),
        num_workers=int(opt.num_worker),
    )
    return dataloader
