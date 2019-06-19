# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform


class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_list = [ele for ele in os.listdir(self.data_dir) if "img" in ele]
        self.transform = transform
        self.cur_img_name = None

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        self.cur_img_name = self.img_list[idx]
        img_path = os.path.join(self.data_dir, self.img_list[idx])
        mask_path = os.path.join(self.data_dir, self.img_list[idx][:8] + "_mask.bmp")
        image = io.imread(img_path) / 255.0
        images = np.stack((image, image, image), axis=-1).astype(np.float32)

        if os.path.exists(mask_path):
            mask = io.imread(mask_path)
            masks = np.zeros((2, mask.shape[0], mask.shape[1]), np.float32)
            masks[0, :, :] = (mask == 127)
            masks[1, :, :] = (mask == 254)
        else:
            masks = []

        if self.transform:
            images = self.transform(images)

        return [images, masks]
