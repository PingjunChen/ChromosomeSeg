# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import mahotas
from skimage import io, filters
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
import cv2
import pickle, json, uuid
import matplotlib.pyplot as plt

from pydaily import filesystem
from pycontour import img, cv2_transform
from config import CHROMOSOME_POS_DICT

import warnings
warnings.filterwarnings("ignore")


class ZernikeMoments:
    def __init__(self, radius):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius)


def extract_karyotypes(karyotype_path):
    img = io.imread(karyotype_path)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    img_fullname = os.path.basename(karyotype_path)
    img_dir = os.path.dirname(karyotype_path)
    img_name = os.path.splitext(img_fullname)[0].replace(" ", "")
    karyotype_dir = os.path.join(img_dir, img_name)
    filesystem.overwrite_dir(karyotype_dir)


    # print("Processing {}".format(img_name))ggg
    for key in CHROMOSOME_POS_DICT.keys():
        cur_h_start = CHROMOSOME_POS_DICT[key]['h_start']
        cur_w_start = CHROMOSOME_POS_DICT[key]['w_start']
        cur_h_end = CHROMOSOME_POS_DICT[key]['h_end']
        cur_w_end = CHROMOSOME_POS_DICT[key]['w_end']
        sub_img = img[cur_h_start:cur_h_end, cur_w_start:cur_w_end]
        cur_chromosome_path = os.path.join(karyotype_dir, key + ".bmp")
        io.imsave(cur_chromosome_path, sub_img)


def extract_all_karyotypes(karyotype_dir):
    karyotype_list = [ele for ele in os.listdir(karyotype_dir) if ele.endswith(".bmp")]
    for cur_karyotype in karyotype_list:
        cur_karyotype_path = os.path.join(karyotype_dir, cur_karyotype)
        extract_karyotypes(cur_karyotype_path)



def get_chromosome_shape(chromosome_path, thresh=28):
    chromosome_name = os.path.splitext(os.path.basename(chromosome_path))[0]
    karyotype_name = os.path.basename(os.path.dirname(chromosome_path))

    ch_img = io.imread(chromosome_path)
    ch_bin = ch_img > thresh
    ch_bin = binary_fill_holes(ch_bin)
    ch_mask = img_as_ubyte(ch_bin)

    try:
        _, cnts, _ = cv2.findContours(ch_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    except:
        import pdb; pdb.set_trace()

    refine_cnts = list(filter(lambda x: cv2.contourArea(x) > 48, cnts))
    refine_cnts.sort(key=(lambda cnt: np.mean(cnt[:, 0, 0])))
    if not (len(refine_cnts) <=2 and len(refine_cnts) >= 0):
        print("In {}, the number of contours in {} is {}".format(karyotype_name, chromosome_name, len(refine_cnts)))
        return None

    chromosome_desc = {}
    desc = ZernikeMoments(21)
    for cur_cnt in refine_cnts:
        cur_arr = cv2_transform.cv_cnt_to_np_arr(cur_cnt)
        shape_mask = img.build_cnt_mask(cur_arr)

        # cur_arr_min_h = np.min(cur_arr[0])
        # cur_arr_min_w = np.min(cur_arr[1])
        # mask_h, mask_w = shape_mask.shape
        # sub_img = ch_img[cur_arr_min_h:cur_arr_min_h+mask_h,
        #                  cur_arr_min_w:cur_arr_min_w+mask_w]
        # shift_cnt = cur_cnt.copy()
        # shift_cnt[:, 0, 0] -= cur_arr_min_w
        # shift_cnt[:, 0, 1] -= cur_arr_min_h
        # sub_img3 = np.stack((sub_img, sub_img, sub_img), axis=-1)
        # cv2.drawContours(sub_img3, [shift_cnt], 0, [127, 0, 0], 2)
        # io.imsave('boundary.png', sub_img3)

        cur_fea = desc.describe(shape_mask)
        shape_name = chromosome_name + '_' + str(uuid.uuid4())[:8]
        chromosome_desc[shape_name] = list(cur_fea)

    return chromosome_desc


def save_chromosome_contours(chromosome_dir):
    chromosome_list = [os.path.join(chromosome_dir, ele) for ele in os.listdir(chromosome_dir) if "_" not in ele]
    karyotype_name = os.path.basename(chromosome_dir)
    karyotype_fea_desc = {}
    for cur_chromosome in chromosome_list:
        chromosome_desc = get_chromosome_shape(cur_chromosome)
        if chromosome_desc != None:
            karyotype_fea_desc.update(chromosome_desc)

    return karyotype_fea_desc


def save_all_karyotype_cnts(karyotype_dir):
    all_chromosome_dict = {}
    karyotype_list = [ele for ele in os.listdir(karyotype_dir) if os.path.isdir(os.path.join(karyotype_dir, ele))]
    for cur_karyotype in karyotype_list:
        cur_karyotype_path = os.path.join(karyotype_dir, cur_karyotype)
        karyotype_fea_desc = save_chromosome_contours(cur_karyotype_path)
        all_chromosome_dict.update(karyotype_fea_desc)

    return all_chromosome_dict

if __name__ == "__main__":
    test_karyotype_dir = "../data/ChromosomeShape/test/"
    # extract_all_karyotypes(test_karyotype_dir)
    train_karyotype_dir = "../data/ChromosomeShape/train/"
    # extract_all_karyotypes(train_karyotype_dir)

    # train_chromosome_dict = save_all_karyotype_cnts(train_karyotype_dir)
    # train_chromosome_desp_path = "../data/ChromosomeShape/train/chromosome_train_feas.json"
    # with open(train_chromosome_desp_path, 'w') as outfile:
    #     json.dump(train_chromosome_dict, outfile)

    test_chromosome_dict = save_all_karyotype_cnts(test_karyotype_dir)
    test_chromosome_desp_path = "../data/ChromosomeShape/test/chromosome_test_feas01.json"
    with open(test_chromosome_desp_path, 'w') as outfile:
        json.dump(test_chromosome_dict, outfile)
