# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/6/18 17:07
"""

# from torch.utils.data import Dataset
# from torchvision.transforms import transforms
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import os.path as osp
# import glob
# import re
# import torch
#
#
# class GanLoader(Dataset):
#     def __init__(self, args):
#         images_path = Path(args.Gan_data_path)
#         images_list = list(images_path.glob('*.npy'))
#         images_list_str = [str(x) for x in images_list]
#         self.images = images_list_str
#
#     def __getitem__(self, item):
#         image_path = self.images[item]
#         return np.load(image_path)
#
#     def __len__(self):
#         return len(self.images)


# class BaseDataset(object):
#     """
#     Base class of reid dataset
#     """
#
#     def get_imagedata_info(self, data):
#         pids, cams = [], []
#         for _, pid, camid in data:
#             pids += [pid]
#             cams += [camid]
#         pids = set(pids)
#         cams = set(cams)
#         num_pids = len(pids)
#         num_cams = len(cams)
#         num_imgs = len(data)
#         return num_pids, num_imgs, num_cams
#
#     def print_dataset_statistics(self):
#         raise NotImplementedError
#
#     @property
#     def images_dir(self):
#         return None
#
#
# class BaseImageDataset(BaseDataset):
#     def print_dataset_statistics(self, train, query, gallery):
#         num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
#         num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
#         num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
#
#         print("Dataset statistics:")
#         print("  ----------------------------------------")
#         print("  subset   | # ids | # images | # cameras")
#         print("  ----------------------------------------")
#         print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
#         print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
#         print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
#         print("  ----------------------------------------")
#
#
# class Market1501(BaseImageDataset):
#     dataset_dir = 'Market1501'
#
#     # **kwargs:一些不确定的参数
#     def __init__(self, root, verbose=True, **kwargs):
#         super(Market1501, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
#         self.query_dir = osp.join(self.dataset_dir, 'query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
#
#         self._check_before_run()
#
#         train = self._process_dir(self.train_dir, relabel=True)
#         query = self._process_dir(self.query_dir, relabel=False)
#         gallery = self._process_dir(self.gallery_dir, relabel=False)
#
#         if verbose:
#             print("=> Market1501 loaded")
#             self.print_dataset_statistics(train, query, gallery)
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#
# 	# 判断文件夹的路径是否存在问题
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))
#
# 	# 获取图片的路径，标注信息（person id，camera id）和图片数量
#     def _process_dir(self, dir_path, relabel=False):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))# 获取.jpg类型的文件
#         pattern = re.compile(r'([-\d]+)_c(\d)')#
#
#         pid_container = set() # 存取训练集中的id，set()有去重功能
#         for img_path in img_paths:
#             pid, _ = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             pid_container.add(pid)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)} # id重排，成为映射关系{...:...,1500:750}
#
#         dataset = []
#         for img_path in img_paths:
#             pid, camid = map(int, pattern.search(img_path).groups())
#             if pid == -1: continue  # junk images are just ignored
#             assert 0 <= pid <= 1501  # pid == 0 means background # 判断pid是否在该范围内
#             assert 1 <= camid <= 6
#             camid -= 1  # index starts from 0，归化到[0,5]
#             if relabel:
#             	pid = pid2label[pid]
#             dataset.append((img_path, pid, camid))
#
#         return dataset
#
#
# class Preprocessor(Dataset):
#     def __init__(self, dataset, root=None, transform=None):
#         super(Preprocessor, self).__init__()
#         self.dataset = dataset
#         self.root = root
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, indices):
#         return self._get_single_item(indices)
#
#     def _get_single_item(self, index):
#         fname, pid, camid = self.dataset[index] # index：0-图片数
#         fpath = fname
#         if self.root is not None:
#             fpath = osp.join(self.root, fname)
#
#         img = Image.open(fpath).convert('RGB') # 读取图片并转换为RGB
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, fname, pid, camid, index


# if __name__ == "__main__":
#     images_path = Path("../Market1501/bounding_box_train")
#     images_list = list(images_path.glob('*.jpg'))
#     images_list_str = [str(x) for x in images_list]
#     i = 0
#     transform = transforms.Compose([
#             transforms.Resize([128, 128]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean=0 std=1
#         ])
#     for path in tqdm(images_list_str):
#         img = Image.open(path)
#         img = transform(img)
#         np.save("../Market1501/pre_load/"+str(i)+".npy", img)
#         i += 1
    # data = torch.utils.data.DataLoader(Preprocessor(Market1501("../").train), batch_size=32)
    # for i, hh in enumerate(data):
    #     print(hh)



import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from utils import *

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
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
