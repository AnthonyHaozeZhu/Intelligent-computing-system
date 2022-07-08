# -*- coding: UTF-8 -*-
"""
@Project ：gen_img 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/6/25 10:34
"""

import argparse
import os

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision import utils as vutils

from model import *


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model", type=str)
    parser.add_argument("--model", default="generator-epoch199.pt", type=str)
    parser.add_argument("--save_dir", default="../img", type=str)
    parser.add_argument("--num_pic", default=12000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model)
    generator = Generator(64)
    generator.load_state_dict(torch.load(model_path, map_location=args.device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    reshape_tensor = transforms.Compose([transforms.Resize((128, 64))])

    for i in tqdm(range(args.num_pic)):
        noise = torch.randn((1, 100)).to(args.device)
        img = generator(noise)
        img = reshape_tensor(img)
        path = os.path.join(args.save_dir, str(i)+".png")
        vutils.save_image(img, path, normalize=True)




