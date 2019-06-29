# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse
import time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from collections import defaultdict
from segnet import UNet, pspnet

from torch.utils.data import DataLoader
from utils import calc_loss, print_metrics
from chromosome_dataset import ChromosomeDataset


def set_args():
    parser = argparse.ArgumentParser(description = 'Chromosome Overlap Segmentation')
    parser.add_argument("--batch_size",      type=int,   default=8,       help="batch size")
    parser.add_argument("--class_num",       type=int,   default=2)
    parser.add_argument("--data_dir",        type=str,   default="../data/OverlapSeg")
    parser.add_argument("--simu_type",       type=str,   default="Fusion")
    parser.add_argument("--model_dir",       type=str,   default="../data/Models/SegModels")
    parser.add_argument("--session",         type=str,   default="r")
    parser.add_argument("--network",         type=str,   default="PSP")
    parser.add_argument("--model_name",      type=str,   default="psp-0.4626.pth")
    parser.add_argument("--gpu",             type=str,   default="4",     help="gpu id")
    args = parser.parse_args()
    return args


def test_seg_model(model, args):
    # prepare dataset
    test_dset = ChromosomeDataset(os.path.join(args.data_dir+args.simu_type, "test_imgs"),
                                  transform = transforms.Compose([transforms.ToTensor(),]))
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics)
        # statistics
        epoch_samples += inputs.size(0)
    print_metrics(metrics, epoch_samples, "test")


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # create model
    model = None
    if args.network == "UNet":
        model = UNet(n_class=args.class_num)
    elif args.network == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(160, 160))
        model.load_pretrained_model(model_path="./segnet/pspnet/pspnet101_cityscapes.caffemodel")
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise Exception("Unknow network: {}".format(args.network))
    print("Net: {} session: {} model name: {}".format(args.network, args.session, args.model_name))
    model_path = os.path.join(args.model_dir, args.simu_type+args.network, args.session, args.model_name)
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    # train model
    test_seg_model(model, args)
