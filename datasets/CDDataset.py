from email.mime import image
import os
import pathlib
import collections
import numpy as np
import torch
import torch.utils.data
import cv2  # pytype: disable=attribute-error
import random
import pandas as pd
import albumentations as A

from glob import glob
from torch.nn.functional import one_hot
from math import ceil
from PIL import Image


class CDDataset(torch.utils.data.Dataset):
    def __init__(self, 
            pre_image_path,
            post_image_path,
            mask_image_path,
            name='LEVIR',
            split='train',
            image_size=128, 
            augmented=False):

        for path in [pre_image_path, post_image_path, mask_image_path]:
            if not os.path.exists(path):
                raise ValueError("Path does not exist: " + path)

        self.pre_images = glob(os.path.join(pre_image_path, "*.png"))
        self.post_images = glob(os.path.join(post_image_path, "*.png"))
        self.mask_images = glob(os.path.join(mask_image_path, "*.png"))

        self.augmented = augmented
        self.image_size = image_size
        self.name = name
        self.split = split
        self.image_size = image_size

        self.base_transform = A.Compose([
            A.RandomCrop(width=image_size, height=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        self.additional_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur()
        ])

        print(f"Pre images: {len(self.pre_images)}")
        print(f"Post images: {len(self.post_images)}")
        print(f"Mask images: {len(self.mask_images)}")

        assert len(self.pre_images) == len(self.post_images)
        assert len(self.pre_images) == len(self.mask_images)
            
    def __getitem__(self, index):
        # load image
        pre_image = cv2.imread(self.pre_images[index])
        post_image = cv2.imread(self.post_images[index])
        mask_image = cv2.imread(self.post_images[index])

        # change to RGB because cv2 read as BGR
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        
        # augmentation
        if self.augmented:
            # base transform
            transformed = self.base_transform(image=pre_image, masks=[post_image, mask_image])
            pre_image = transformed['image']
            post_image = transformed['masks'][0]
            mask_image = transformed['masks'][1]

            # additional transform
            transformed = self.additional_transform(image=pre_image, mask=post_image)
            pre_image = transformed['image']
            post_image = transformed['mask']

        # normalizing and convert to tensor
        pre_image = pre_image.transpose(2, 0, 1).astype('float32') / 255.
        post_image = post_image.transpose(2, 0, 1).astype('float32') / 255.

        # threshold all pixel greater than 0 to 255
        mask_image[mask_image > 0] = 255
        mask_image = mask_image.transpose(2, 0, 1).astype('uint8') / 255
        
        return pre_image, post_image, mask_image
            
    def __len__(self):
        return len(self.pre_images)