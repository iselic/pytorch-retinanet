train_csv = '/content/gdrive/My Drive/Colab Notebooks/Pytorch/data/seedling_test/data/training/training_annotations.csv'  # @param {type:"string"}
val_csv = '/content/gdrive/My Drive/Colab Notebooks/Pytorch/data/seedling_test/data/training/validation_annotations.csv'  # @param {type:"string"}
classes_csv = '/content/gdrive/My Drive/Colab Notebooks/Pytorch/data/seedling_test/data/label_map.csv'  # @param {type:"string"}
output_prefix = 'seedlings'
out_dir = '/content/gdrive/My Drive/Colab Notebooks/Pytorch/data/seedling_test/checkpoints/'

model_fname = None  # '/content/gdrive/My Drive/Colab Notebooks/Pytorch/data/wyn_trees/trees_state_2998.pt'

# @markdown ### Parameters

epochs = 2000  # @param {type:"integer"}
resnet_depth = 18  # @param [18, 34, 50, 101, 152]
use_gpu = True  # @param {type:"boolean"}

# @markdown ---


import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def train(train_csv,val_csv,classes_csv,depth=50,epochs=1000,steps=100,out_dir ='',out_prefix=''):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the data loaders

    dataset_train = CSVDataset(train_file=train_csv, class_list=classes_csv,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    dataset_val = CSVDataset(train_file=val_csv, class_list=classes_csv,
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if resnet_depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # retinanet = torch.load(model_fname)

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    if model_fname is not None:
        retinanet.load_state_dict(torch.load(model_fname))

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    start_time = time.clock()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print('Epoch: {} | Running loss: {:1.5f} | Elapsed Time: {}'.format(epoch_num, np.mean(loss_hist),
                                                                            time.clock() - start_time))

        mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        if (epoch_num) % steps == 0:
            torch.save(retinanet.module, '{}{}_model_{}.pt'.format(out_dir, output_prefix, epoch_num))
            torch.save(retinanet.state_dict(), '{}{}_state_{}.pt'.format(out_dir, output_prefix, epoch_num))

    torch.save(retinanet, out_dir + 'model_final.pt'.format(epoch_num))
    torch.save(retinanet.state_dict(), out_dir + 'state_final_.pt'.format(epoch_num))

if __name__ == '__main__':
	train()