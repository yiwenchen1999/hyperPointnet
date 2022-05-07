# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:36:23 2022

@author: ThinkPad
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from PartialScan import PartialScans
from model import feature_transform_regularizer
from pointnetCls import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset_type', type=str, default='Shapenet_partial', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

latent_code = r"E:\Code\IVL\shapeSearch\HyperPointnet\pointnet\03001627\embed_feats.pickle"
latent_code_test = r"E:\Code\IVL\shapeSearch\HyperPointnet\pointnet\03001627\embed_feats_test.pickle"
shape_folder = r"D:\data\dataset_small_partial\03001627"

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

print("train set lenth: "+ str(len(dataset)) +", test set length: "+ str(len(test_dataset)))
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=2, feature_transform=opt.feature_transform)

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    # scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        print(points)
        # target = target[:, 0]
        # points = points.transpose(2, 1)
        # points, target = points.cuda(), target.cuda()
        # optimizer.zero_grad()
        # classifier = classifier.train()
        # pred, trans, trans_feat = classifier(points)
        # loss = F.nll_loss(pred, target)
        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        # loss.backward()
        # optimizer.step()
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        # if i % 10 == 0:
        #     j, data = next(enumerate(testdataloader, 0))
        #     points, target = data
        #     target = target[:, 0]
        #     points = points.transpose(2, 1)
        #     points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
