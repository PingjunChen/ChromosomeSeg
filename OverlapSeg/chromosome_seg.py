# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse, time, pickle
import cv2
from skimage import io, measure
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


from chromosome_dataset import ChromosomeDataset
from utils import mask2color, calc_loss
from utils import refine_pred
from utils import assign_combine
from utils import ZernikeMoments, cal_assignment_fea
from segnet import UNet


def set_args():
    parser = argparse.ArgumentParser(description = 'Chromosome Overlap Segmentation')
    parser.add_argument("--batch_size",      type=int,   default=1,       help="batch size")
    parser.add_argument("--class_num",       type=int,   default=2,       help="number of category")
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--model_name",      type=str,   default="unet-0.1283.pth")
    parser.add_argument("--lda_model_path",  type=str,   default="lda_model.pkl")
    parser.add_argument("--gpu",             type=str,   default="2",     help="gpu id")
    parser.add_argument("--seed",            type=int,   default=1234,    help="seed")

    args = parser.parse_args()
    return args


def gen_test_dataloader():
    trans = transforms.Compose([transforms.ToTensor()])
    # trans = None
    test_dset = ChromosomeDataset(os.path.join(args.data_dir, "OverlapSeg/test_imgs"), transform = trans)
    test_dloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return test_dloader


def test_model(model, dloader, save_pic=True):
    start_time = time.time()

    desc = ZernikeMoments(radius=21)
    lda_paras = pickle.load(open(args.lda_model_path, "rb"))
    def lda_pred(fea):
        testVector = np.matmul(fea, lda_paras['ProjectMat'])
        class_num = lda_paras['ClassMean'].shape[0]
        tmp_dist = np.matmul(np.ones((class_num, 1)), np.expand_dims(testVector, axis=0))
        dist = np.sum((tmp_dist - lda_paras['ClassMean'])**2, axis=1)
        label, min_dist = np.argmin(dist), min(dist)
        return label, min_dist

    # testing on each image
    metrics = defaultdict(float)
    for ind, (inputs, labels) in enumerate(dloader):
        filename = os.path.splitext(dloader.dataset.cur_img_name)[0]
        save_path = os.path.join(args.data_dir, "Predictions", filename+".png")

        img_ori = inputs[0].cpu().numpy().transpose((1, 2, 0))
        mask = labels[0].cpu().numpy().transpose((1, 2, 0))
        mask = (mask[...,0] * 1 + mask[...,1] * 255).astype(np.uint8)
        mask_c = mask2color(mask)

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = calc_loss(outputs, labels, metrics)

        preds = torch.sigmoid(outputs)
        preds = preds.data.cpu().numpy()
        pred = preds[0].transpose((1, 2, 0))

        # overlap override single
        single = pred[...,0] > 0.5
        overlap = pred[...,1] > 0.5

        # basic combine
        combine1 = np.zeros_like(single, dtype=np.uint8)
        combine1[single==True] = 1
        combine1[overlap==True] = 255
        pred_c1 = mask2color(combine1)
        single = combine1 == True

        # combine prediction
        combine = refine_pred(single, overlap)
        pred_c = mask2color(combine)

        if (ind+1) % 20 == 0:
            print("Processing {}/{}".format(ind+1, len(dloader)))
        if save_pic == True:
            # with PdfPages(save_path) as pdf:
            fig=plt.figure(figsize=(10, 3))
            fig.add_subplot(1, 3, 1)
            plt.imshow(img_ori)
            plt.title("Input image")
            plt.axis('off')
            fig.add_subplot(1, 3, 2)
            plt.imshow(mask_c)
            plt.title("Ground-truth")
            plt.axis('off')
            fig.add_subplot(1, 3, 3)
            plt.imshow(pred_c1)
            plt.title("Prediction")
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()

    avg_dice_single = metrics["dice_single"] / len(dloader)
    avg_dice_overlap = metrics["dice_overlap"] / len(dloader)

    elapsed_time = time.time() - start_time
    print("Takes {} seconds on {} images".format(elapsed_time, len(dloader.dataset)))
    print("Average single chromosome dice ratio is: {}".format(avg_dice_single))
    print("Average overlap chromosome dice ratio is: {}".format(avg_dice_overlap))


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch.backends.cudnn as cudnn
    torch.backends.cudnn.deterministic=True
    cudnn.benchmark = True

    # load model
    model = UNet(n_class=args.class_num)
    model_path = os.path.join(args.data_dir, "Models/SegModels/AddUNet/s4", args.model_name)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    # prepare dataset
    dloader = gen_test_dataloader()
    # test model
    test_model(model, dloader)
