# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import mahotas
import itertools
from skimage import filters
from skimage import img_as_ubyte
from skimage import morphology, measure

from scipy.ndimage import binary_fill_holes
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from pycontour import img, cv2_transform


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))

    return dice.mean(dim=0)


def calc_loss(pred, target, metrics, bce_weight=0.2):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + (1.0 - dice.mean()) * (1.0 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_mean'] += dice.mean().data.item() * target.size(0)
    metrics['dice_single'] += dice[0].data.item() * target.size(0)
    metrics['dice_overlap'] += dice[1].data.item() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


def mask2color(mask):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56),
                         (0, 0, 117), (128, 128, 0), (191, 239, 69), (145, 30, 180)])
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    color_img[mask==255] = colors[0]
    for ic in np.arange(1, len(colors)):
        color_img[mask==ic] = colors[ic]

    return color_img.astype(np.uint8)



def refine_seg(seg):
    fill_seg = binary_fill_holes(seg)
    clean_seg = morphology.remove_small_objects(fill_seg, min_size=32)

    return clean_seg


def remove_weak_connection(bin_img):
    final_img = np.zeros_like(bin_img, dtype=bool)
    all_labels = measure.label(bin_img)
    region_num_ttl = len(np.unique(all_labels))

    for r in np.arange(1, region_num_ttl):
        cur_region = (all_labels == r)
        open_region = morphology.opening(cur_region, morphology.square(3))
        # open_region = morphology.opening(cur_region, morphology.square(2))
        region_num = len(np.unique(measure.label(open_region))) - 1
        if region_num > 1:
            final_img += open_region
            diff = cur_region ^ open_region
            diff_labels = measure.label(diff)
            diff_num = len(np.unique(diff_labels))
            for d in np.arange(1, diff_num):
                cur_diff = diff_labels == d
                diff_add = open_region + cur_diff
                cur_region_num = len(np.unique(measure.label(diff_add))) - 1
                if  cur_region_num == region_num:
                    final_img += cur_diff
        else:
            final_img += cur_region
    return final_img


def refine_pred(single, overlap):
    combine = np.zeros_like(single, dtype=np.uint8)

    # Remove weak connnection in single prediction
    single = remove_weak_connection(single)
    single = morphology.remove_small_objects(single, min_size=32)

    combine = measure.label(single)
    overlap = binary_fill_holes(overlap)
    combine[overlap==True] = 255

    return combine


def check_single_region(combine):
    bin = combine > 0
    label_num = len(np.unique(measure.label(bin))) - 1

    single_flag = label_num == 1
    return single_flag

def check_two_chromosomes(ch1, ch2):
    flag1 = check_single_region(ch1)
    flag2 = check_single_region(ch2)

    if flag1 == False or flag2 == False:
        return False

    return True


def assign_combine(combine):
    all_labels = measure.label(combine)
    single_num = len(np.unique(all_labels)) - 2
    overlap = combine == 255
    assignments = []

    if single_num == 2:
        ch1 = (combine == 1) + overlap
        ch2 = (combine == 2) + overlap
        if check_two_chromosomes(ch1, ch2) == False:
            return None
        _, cnts1, _ = cv2.findContours(img_as_ubyte(ch1), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        _, cnts2, _ = cv2.findContours(img_as_ubyte(ch2), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        if len(cnts1) != 1 or len(cnts2) != 1:
            return None
        else:
            assignments.append([cnts1[0], cnts2[0]])
    if single_num == 3:
        labels = np.arange(1, single_num+1)
        for label in labels:
            ch1, ch2 = overlap.copy(), overlap.copy()
            ch1 += (combine == label)
            rest_labels = [x for x in labels if x != label]
            for rest in rest_labels:
                ch2 += (combine == rest)
            if check_two_chromosomes(ch1, ch2) == False:
                return None
            _, cnts1, _ = cv2.findContours(img_as_ubyte(ch1), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            _, cnts2, _ = cv2.findContours(img_as_ubyte(ch2), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(cnts1) != 1 or len(cnts2) != 1:
                return None
            else:
                assignments.append([cnts1[0], cnts2[0]])
    if single_num == 4:
        all_possibles = list(itertools.combinations(np.arange(1, 5).tolist(), 2))
        half_len = int(len(all_possibles) / 2)
        for i in np.arange(half_len):
            labels1 = all_possibles[i]
            labels2 = all_possibles[len(all_possibles) - i - 1]
            ch1, ch2 = overlap.copy(), overlap.copy()
            for ele in labels1:
                ch1 += (combine == ele)
            for ele in labels2:
                ch2 += (combine == ele)
            if check_two_chromosomes(ch1, ch2) == False:
                return None
            _, cnts1, _ = cv2.findContours(img_as_ubyte(ch1), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            _, cnts2, _ = cv2.findContours(img_as_ubyte(ch2), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(cnts1) != 1 or len(cnts2) != 1:
                return None
            else:
                assignments.append([cnts1[0], cnts2[0]])
    return assignments


def cal_assignment_fea(assignment, desc):
    cnt1, cnt2 = assignment[0], assignment[1]
    arr1 = cv2_transform.cv_cnt_to_np_arr(cnt1)
    arr2 = cv2_transform.cv_cnt_to_np_arr(cnt2)
    mask1 = img.build_cnt_mask(arr1)
    mask2 = img.build_cnt_mask(arr2)
    fea1 = desc.describe(mask1)
    fea2 = desc.describe(mask2)

    return fea1, fea2


class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)
