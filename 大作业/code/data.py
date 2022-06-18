# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/6/18 17:07
"""

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image


class GanLoader(Dataset):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.Resize([args.img_w, args.img_h]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean=0 std=1
        ])
        images_path = Path(args.Gan_data_path)
        images_list = list(images_path.glob('*.jpg'))
        images_list_str = [str(x) for x in images_list]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        return self.transform(image)

    def __len__(self):
        return len(self.images)

