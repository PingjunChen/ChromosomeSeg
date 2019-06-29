# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import random, uuid
import matplotlib.pyplot as plt
from pydaily import filesystem


def plot_lr(l_img, r_img, l_title=None, r_title=None, suptitle=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    axes[0].imshow(l_img, cmap="gray")
    axes[0].set_title(l_title)
    axes[0].set_axis_off()

    axes[1].imshow(r_img, cmap="gray")
    axes[1].set_title(r_title)
    axes[1].set_axis_off()

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()



def simulate_overlap(dataset_dir, cur_id, overlap_size, simulate_num,
                     overlap_ratio_low=0.5, overlap_ratio_high=1.0):
    overlap_dir = os.path.join(dataset_dir, "simu_train")
    if not os.path.exists(overlap_dir):
        os.makedirs(overlap_dir)

    img_list = os.listdir(os.path.join(dataset_dir, cur_id+"use"))
    mask_list = os.listdir(os.path.join(dataset_dir, cur_id+"mask"))
    img_list.sort()
    mask_list.sort()

    success_num, try_num = 0, 0
    while success_num < simulate_num:
        try_num += 1

        if try_num > 5000:
            print("Try more than {} times, quit!".format(try_num))
            break

        cur_selection = random.sample(img_list, 2)
        img1_path = os.path.join(dataset_dir, cur_id+"use", cur_selection[0])
        img2_path = os.path.join(dataset_dir, cur_id+"use", cur_selection[1])
        mask1_path = os.path.join(dataset_dir, cur_id+"mask", cur_selection[0])
        mask2_path = os.path.join(dataset_dir, cur_id+"mask", cur_selection[1])

        overlap_img = np.zeros((overlap_size, overlap_size), np.float32)
        overlap_mask = np.zeros((overlap_size, overlap_size), np.uint8)

        img1 = io.imread(img1_path)
        h1, w1 = img1.shape[0], img1.shape[1]
        img2 = io.imread(img2_path)
        h2, w2 = img2.shape[0], img2.shape[1]

        max_size = overlap_size - 3
        if h1 > max_size or w1 > max_size or h2 > max_size or w2 > max_size:
            continue

        rand_h1 = random.choice(np.arange(overlap_size-h1))
        rand_w1 = random.choice(np.arange(overlap_size-w1))
        rand_h2 = random.choice(np.arange(overlap_size-h2))
        rand_w2 = random.choice(np.arange(overlap_size-w2))

        mask1 = io.imread(mask1_path) / 2
        mask1 = mask1.astype(np.uint8)
        mask2 = io.imread(mask2_path) / 2
        mask2 = mask2.astype(np.uint8)
        overlap_mask[rand_h1:rand_h1+h1, rand_w1:rand_w1+w1] += mask1
        overlap_mask[rand_h2:rand_h2+h2, rand_w2:rand_w2+w2] += mask2

        num_overlap = np.count_nonzero(overlap_mask == 254)
        num_single = np.count_nonzero(overlap_mask == 127)
        overlap_ratio = num_overlap * 2.0 / (num_single + num_overlap * 2.0)

        if overlap_ratio > overlap_ratio_low and overlap_ratio < overlap_ratio_high:
            extend_img1 = np.zeros((overlap_size, overlap_size), np.uint8)
            extend_img1[rand_h1:rand_h1+h1, rand_w1:rand_w1+w1] = img1
            overlap_img += extend_img1
            extend_img2 = np.zeros((overlap_size, overlap_size), np.uint8)
            extend_img2[rand_h2:rand_h2+h2, rand_w2:rand_w2+w2] = img2
            overlap_img += extend_img2
            sum_overlap_img = overlap_img.copy()
            sum_overlap_img[sum_overlap_img > 255] = 255
            sum_overlap_img = sum_overlap_img.astype(np.uint8)

            overlap_r = overlap_mask == 254
            overlap_img[overlap_r] = 0
            inv_overlap_r = np.invert(overlap_r)
            extend_img1[inv_overlap_r] = 0
            extend_img2[inv_overlap_r] = 0
            overlap_ratio1 = round(np.random.uniform(0.6, 1.0), 2)
            overlap_ratio2 = round(np.random.uniform(0.6, 1.0), 2)
            rand_overlap_img = overlap_img.copy()
            rand_overlap_img += overlap_ratio1 * extend_img1
            rand_overlap_img += overlap_ratio2 * extend_img2
            rand_overlap_img[rand_overlap_img > 255] = 255
            rand_overlap_img = rand_overlap_img.astype(np.uint8)

            l_title = "Summing overlap"
            r_title = r"Overlap with $ \alpha_{1}= $" + "{}".format(overlap_ratio1) + r" and $ \alpha_{2}= $" + "{}".format(overlap_ratio2)
            plot_lr(sum_overlap_img, rand_overlap_img, l_title, r_title)

        else:
            continue

if __name__ == "__main__":
    dataset_dir = "../data/OverlapSeg/single_chromosomes"
    img_ids = [str(id) for id in np.arange(1, 80)]
    for cur_id in img_ids:
        print("Processing {}".format(cur_id))
        # simulate_overlap(dataset_dir, cur_id, overlap_size=160, simulate_num=10, overlap_ratio_low=0.3, overlap_ratio_high=0.5)
        simulate_overlap(dataset_dir, cur_id, overlap_size=160, simulate_num=10, overlap_ratio_low=0.1, overlap_ratio_high=0.3)
