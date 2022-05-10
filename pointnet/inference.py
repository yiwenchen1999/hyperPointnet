# -*- coding: utf-8 -*-
"""
Created on Tue May 10 04:27:29 2022

@author: ThinkPad
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from PartialScan import PartialScans,unpickle
from model import feature_transform_regularizer
from pointnetCls import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=3, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--checkpoint', type=str, default=r'E:\Code\IVL\shapeSearch\HyperPointnet\checkpoint.pth', help="checkpoint dir")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

latent_code = r"E:\Code\IVL\shapeSearch\HyperPointnet\pointnet\03001627\ocnet_shapefeature_pc\embed_feats_train.pickle"
latent_code_test = r"E:\Code\IVL\shapeSearch\HyperPointnet\pointnet\03001627\ocnet_shapefeature_pc\embed_feats_test.pickle"
shape_folder = r"/gpfs/data/ssrinath/ychen485/partialPointCloud/03001627"
latent_dim = 512

dataset = PartialScans(latentcode_dir = latent_code, shapes_dir = shape_folder)

test_dataset = PartialScans(latentcode_dir = latent_code_test, shapes_dir = shape_folder)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

latent_dict = unpickle(latent_code)
keylist = list(latent_dict.keys())
latent_dict_test = unpickle(latent_code_test)
keylist_test = list(latent_dict_test.keys())

print("train set lenth: "+ str(len(dataset)) +", test set length: "+ str(len(test_dataset)))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=2, feature_transform=opt.feature_transform)

if opt.checkpoint != " ":
    checkpoint = torch.load(opt.checkpoint)
    classifier.load_state_dict(checkpoint)
