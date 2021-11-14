import os
from datasets.dataloader_assd import *
from sklearn.model_selection import StratifiedKFold


class getter_dataloader(object):
    """ Choose datasets. """

    def __init__(self, opt):
        self.opt = opt
        dataset = self.opt.data
        enable_data_filtered = self.opt.enable_data_filtered

        if dataset == 'assd':
            dir_img = '_data/assd/original_images/'
            dir_gt = '_data/assd/label_images_semantic/'
            filename_img = []
            for _, _, file_list in os.walk(dir_img):
                for track_name in file_list:
                    filename_img.append(dir_img + track_name[:-4])
            self.filename_img = filename_img
            self.get_dataset_dataloader = get_assd_dataloader
        else:
            raise RuntimeError('Dataset ' + dataset + ' not found!')

        self.data_splitter = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    def get(self, fold):
        """
        Return
        """
        assert 0 <= fold <= 9
        for i, (train_index, val_index) in enumerate(self.data_splitter.split(self.filename_img, self.filename_img)):
            if i != fold:
                continue
            else:
                train_loader, val_loader = self.get_dataset_dataloader(self.opt,
                                                                       [self.filename_img[i] for i in train_index],
                                                                       [self.filename_img[i] for i in val_index])
                return train_loader, val_loader


def get_num_label(dataset: str):
    if dataset == 'ASSD':
        return 23
    else:
        raise RuntimeError('Dataset ' + dataset + ' not found!')
