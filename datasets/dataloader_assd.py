import os
import argparse
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ASSD(Dataset):
    """ Event stream datasets. """
    def __init__(self,
                 list_filename: list,
                 mean,
                 std,
                 augment_hvflip: float,
                 augment_wgn: float,
                 root: str = '_data/assd/'):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        if std is None:
            std = [0.3169534, 0.31788042, 0.31566283]
        self._walker = list_filename
        self.length = len(self._walker)
        self.path_img = os.path.join(root, 'original_images')
        self.path_gt = os.path.join(root, 'label_images_semantic')
        self.ext_img = '.jpg'
        self.ext_gt = '.png'

        self.augment_hvflip = augment_hvflip
        self.augment_wgn = augment_wgn

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.h_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.v_flip = transforms.RandomVerticalFlip(p=1.0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        img = self.train_transform(np.array(Image.open(self._walker[index] + self.ext_img)) / 255)
        path_gt = self._walker[index][:11] + 'label_images_semantic' + self._walker[index][-4:] + self.ext_gt
        gt = torch.Tensor(np.array(Image.open(path_gt)))

        # Augmentation
        if np.random.rand() < self.augment_hvflip:
            p = np.random.rand()
            if p < 1/3:
                img = self.h_flip(img)
                gt = self.h_flip(gt)
            elif p < 2/3:
                img = self.v_flip(img)
                gt = self.v_flip(gt)
            else:
                img = self.h_flip(self.v_flip(img))
                gt = self.h_flip(self.v_flip(gt))

        if self.augment_wgn > 0:
            img += torch.randn(img.shape) * self.augment_wgn

        return img.float(), gt.float()


def get_assd_dataloader(opt: argparse.Namespace, train_list: list, val_list: list, test_list: list):
    """ Load data and prepare dataloader. """
    # Calculate mean and std
    if opt.recalculate_mean_std:
        mean, std = get_mean_std(train_list)
    else:
        mean = [0.64917991, 0.64219542, 0.61161699]
        std = [0.20462199, 0.20038003, 0.2113568]

    # Instancelize datasets
    train_data = ASSD(list_filename=train_list, mean=mean, std=std, augment_hvflip=opt.enable_hvflip,
                      augment_wgn=opt.enable_hvflip)

    val_data = ASSD(list_filename=val_list, mean=mean, std=std, augment_hvflip=0, augment_wgn=0)

    test_data = ASSD(list_filename=test_list, mean=mean, std=std, augment_hvflip=0, augment_wgn=0)

    # Instancelize dataloader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    test_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def get_mean_std(train_list):
    """
    Calculate mean and std from given filename list
    """
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in tqdm(range(len(train_list)), desc='- (Calculating mean and std)   ', leave=False):
        fn = train_list[i]
        img = np.array(Image.open(fn + '.jpg')) / 255
        for ch in range(img.shape[-1]):
            img_ch = img[:, :, ch]
            mean[ch] += img_ch.mean()
            std[ch] += img_ch.std()

    return mean / len(train_list), std / len(train_list)



