import pytorch_lightning as pl
import os
import pandas as pd
import cv2
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets.CDDataset import CDDataset

class CDDataModule(pl.LightningDataModule):
    def __init__(self, 
            root_data_path: str = "datasets/LEVIR", 
            pre_image_dir: str = "A",
            post_image_dir: str = "B",
            mask_image_dir: str = "label",
            batch_size: int = 32, 
            num_workers: int = 8):
        super().__init__()

        self.root_data_path = root_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_image_dir = pre_image_dir
        self.post_image_dir = post_image_dir
        self.mask_image_dir = mask_image_dir

    def setup(self, stage = None):
        print(f'setup: {self.root_data_path}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = CDDataset(
                                root_data_path = self.root_data_path, 
                                pre_image_dir = self.pre_image_dir,
                                post_image_dir = self.post_image_dir,
                                mask_image_dir = self.mask_image_dir,
                                augmented=True)
            
            self.val_set   = CDDataset(root_data_path = self.root_data_path, 
                                pre_image_dir = self.pre_image_dir,
                                post_image_dir = self.post_image_dir,
                                mask_image_dir = self.mask_image_dir,)

        if stage == "validate" or stage is None:
            self.val_set   = CDDataset(root_data_path = self.root_data_path, 
                                pre_image_dir = self.pre_image_dir,
                                post_image_dir = self.post_image_dir,
                                mask_image_dir = self.mask_image_dir,)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set   = CDDataset(root_data_path = self.root_data_path, 
                                pre_image_dir = self.pre_image_dir,
                                post_image_dir = self.post_image_dir,
                                mask_image_dir = self.mask_image_dir,)

        if stage == "predict" or stage is None:
            self.predict_set   = CDDataset(root_data_path = self.root_data_path, 
                                pre_image_dir = self.pre_image_dir,
                                post_image_dir = self.post_image_dir,
                                mask_image_dir = self.mask_image_dir,)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)