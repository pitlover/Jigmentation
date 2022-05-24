from typing import Dict, Tuple, Optional
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
import torch.nn as nn
import torch.nn.functional as F  # noqa
import types, collections


class SegDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 data_type: str = "PASCAL",
                 mode: str = "train",
                 img_size: Optional[Tuple[int, int]] = None,
                 ):
        super().__init__()

        mode = mode.lower()
        if mode not in ("train", "val", "test"):
            raise ValueError(f"SegDataset mode {mode} is not supported.")
        data_type = data_type.upper()
        if data_type not in ("PASCAL", "ADE20K"):
            raise ValueError(f"SegDataset data_type {data_type} is not supported.")

        self.data_path = data_path
        self.data_type = data_type
        self.mode = mode

        # ------------------------ PASCAL ------------------------------#
        if self.data_type == "PASCAL":
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]

            self.train_transforms = Compose([
                Resize(img_size),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                ToTensor(),
                Normalize(mean_vals, std_vals)
            ])
            self.eval_transforms = Compose([
                Resize(img_size),
                ToTensor(),
                Normalize(mean_vals, std_vals)
            ])

            if mode == "train":
                with open("./dataset/pascal/train_aug.txt", 'r') as f:
                    self.filenames = list(f.readlines())
            elif mode == "val":
                with open("./dataset/pascal/val.txt", 'r') as f:
                    self.filenames = list(f.readlines())
            elif mode == "test":
                with open("./dataset/pascal/test.txt", 'r') as f:
                    self.filenames = list(f.readlines())

            self.height, self.width = img_size[0], img_size[1]
            self.img_path = os.path.join(self.data_path, "JPEGImages")
            self.gt_path = os.path.join(self.data_path, "SegmentationClassAug")

        else:  # should not be here
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict:
        path = self.filenames[idx]

        image_path = os.path.join(self.img_path, path)
        gt_path = os.path.join(self.gt_path, path)

        # ------------------------ train ------------------------------#
        if self.data_type == "PASCAL":
            image = Image.open(image_path).convert('RGB')

            if self.mode == "train":
                image = self.train_transforms(image)
            elif self.mode == "val" or "test":
                image = self.eval_transforms(image)

            gt = Image.open(gt_path)  # (h, w), 16-bit grayscale


        sample = {'image': image, 'gt': gt,
                  'image_path': image_path, 'gt_path': gt_path}

        return sample


class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, (self.size, self.size), self.interpolation)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)
