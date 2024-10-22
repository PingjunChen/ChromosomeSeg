# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import shutil


def gen_multiple_copy(img_dir, save_dir, copy):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = [ele for ele in os.listdir(img_dir) if "img" in ele]
    for ele in img_list:
        ori_filename = ele[:8]
        ori_img_name = ori_filename + "_img.bmp"
        ori_mask_name = ori_filename + "_mask.bmp"

        for ind in np.arange(1, copy+1):
            cur_ind = str(ind)
            new_img_name = cur_ind + ori_img_name[len(cur_ind):]
            new_mask_name = cur_ind + ori_mask_name[len(cur_ind):]
            shutil.copyfile(os.path.join(img_dir, ori_img_name), os.path.join(save_dir, new_img_name))
            shutil.copyfile(os.path.join(img_dir, ori_mask_name), os.path.join(save_dir, new_mask_name))

if __name__ == "__main__":
    train_img_dir = "../data/OverlapSegFusion/r_train_imgs"
    save_img_dir = "../data/OverlapSegFusion/train_imgs"
    gen_multiple_copy(train_img_dir, save_img_dir, copy=9)
