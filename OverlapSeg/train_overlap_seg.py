# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse
import time, copy
import torch
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
    parser.add_argument("--lr",              type=float, default=1.0e-4,  help="learning rate (default: 0.0001)")
    parser.add_argument("--maxepoch",        type=int,   default=32,      help="number of epochs to train")
    parser.add_argument("--decay_epoch",     type=int,   default=10,      help="lr start to decay linearly from decay_epoch")
    parser.add_argument("--display_freq",    type=int,   default=50,      help="plot the results every {} batches")
    parser.add_argument("--save_freq",       type=int,   default=1,       help="how frequent to save the model")
    parser.add_argument("--class_num",       type=int,   default=2)
    parser.add_argument("--data_dir",        type=str,   default="../data/OverlapSeg")
    parser.add_argument("--model_dir",       type=str,   default="../data/Models")
    parser.add_argument("--network",         type=str,   default="UNet")
    parser.add_argument("--gpu",             type=str,   default="7",     help="gpu id")
    parser.add_argument("--session",         type=str,   default="s9",     help="training session")
    parser.add_argument("--seed",            type=int,   default=1234,    help="training seed")

    args = parser.parse_args()
    return args



def gen_dataloader(args):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dset = ChromosomeDataset(os.path.join(args.data_dir, args.session+"_train_imgs"), transform = trans)
    val_dset = ChromosomeDataset(os.path.join(args.data_dir, "val_imgs"), transform = trans)

    dataloaders = {
        'train': DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }

    return dataloaders


def train_model(model, args):
    dataloaders = gen_dataloader(args)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.1)

    best_loss = 1e10
    for epoch in np.arange(1, args.maxepoch + 1):
        print('Epoch {}/{}'.format(epoch, args.maxepoch))
        print('-' * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                # print("saving best model")
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Epoch {:2d} takes {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))

    print("================================================================================")
    print("Training finished...")
    best_loss_str = '{:.4f}'.format(best_loss)
    print('Best val loss: ' + best_loss_str)
    # Save best model
    best_model_dir =  os.path.join(args.model_dir, "SegModels", args.network, args.session)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_name = "unet-" + str(best_loss_str) + ".pth"
    best_model_path = os.path.join(best_model_dir, best_model_name)
    torch.save(best_model, best_model_path)



if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # create model
    model = None
    if args.network == "UNet":
        model = UNet(n_class=args.class_num)
    elif args.model_name == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(160, 160))
        model.load_pretrained_model(model_path="./segnet/pspnet/pspnet101_cityscapes.caffemodel")
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise Exception("Unknow network: {}".format(args.network))
    model.cuda()
    # prepare dataset

    # train model
    train_model(model, args)
