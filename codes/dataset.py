import os
from glob import glob
import random

from PIL import Image, ImageOps

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

coco_transforms = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize(   (0.4984, 0.5092, 0.5008),
                            (0.2316, 0.2321, 0.2332))
            ])

coco_reverse_transforms = transforms.Compose([
    transforms.Normalize(   (-0.4984/0.2316, -0.5092/0.2321, -0.5008/0.2332),
                            (1/0.2316, 1/0.2321, 1/0.2332)),
])

bn_transforms = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor()
            ])    
to_tensor_transforms = transforms.Compose([
            transforms.ToTensor()
            ])    

class drive(Dataset):
    '''train or test mode'''
    def __init__(self, root='../../Datasets/DRIVE/', mode='train', img_type='RGB'):
        self.root = root
        self.mode = mode
        self.img_type = img_type
        if self.img_type == 'RGB':
            self.transform = coco_transforms
            self.reverse_transform = coco_reverse_transforms
        elif self.img_type == 'L':
            self.transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.ToTensor(),
                transforms.Normalize(   0.4502,
                                        0.2225)
                ])
            self.reverse_transform = transforms.Compose([
                transforms.Normalize(   -0.4502/0.2225,
                                        1/0.2225),
                ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor()
            ])   

        self.images = sorted(glob(f'{self.root}{self.mode}/images/*.jpg'))
        self.masks = sorted(glob(f'{self.root}{self.mode}/mask/*.jpg'))
        if self.mode == 'train':
            self.gts = sorted(glob(f'{self.root}{self.mode}/1st_manual/*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        orig = self.transform(Image.open(f'{self.images[index]}').convert(self.img_type))
        mask = self.mask_transform(Image.open(f'{self.masks[index]}').convert('1'))
        if self.mode == 'train':
            gt = self.transform(Image.open(f'{self.gts[index]}').convert('1'))
            return orig, mask, gt
        else:
            return orig, mask, mask

    def __detransform__(self, tensor, orig=False):
        
        tensor = self.reverse_transform(tensor)

        minFrom = tensor.min()
        maxFrom = tensor.max()
        tensor = (tensor - minFrom) / (maxFrom - minFrom)

        if orig == True:
            return 1 - tensor
        else:
            return tensor