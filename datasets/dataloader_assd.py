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
                 mean: np.ndarray,
                 std: np.ndarray,
                 augment: bool = False,
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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        img = np.array(Image.open(self.path_img + self._walker[index] + self.ext_img)) / 255
        gt = np.array(Image.open(self.path_gt + self._walker[index] + self.ext_gt)) / 255
        return self.transform(img), self.transform(gt)


def get_assd_dataloader(opt: argparse.Namespace, train_list: list, val_list: list):
    """ Load data and prepare dataloader. """
    # Calculate mean and std
    mean, std = get_mean_std(train_list)

    # Instancelize datasets
    train_data = ASSD(list_filename=train_list, augment=True, mean=mean, std=std)

    val_data = ASSD(list_filename=val_list, augment=False, mean=mean, std=std)

    # Instancelize dataloader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader


def get_mean_std(train_list):
    """
    Calculate mean and std from given filename list
    """
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in tqdm(range(len(train_list)), desc='- (Calculating mean and std)   ', leave=False):
        fn = train_list[i]
        # img = np.array(Image.open('_data/assd/original_images/' + fn + '.jpg')) / 255
        img = np.array(Image.open('_data/assd/original_images/' + fn + '.jpg')) / 255
        for ch in range(img.shape[-1]):
            img_ch = img[:, :, ch]
            mean[ch] += img_ch.mean()
            mean[ch] += img_ch.std()

    return mean / len(train_list), std / len(train_list)



