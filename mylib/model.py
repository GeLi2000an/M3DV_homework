#导入相关模块
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import csv
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool3d(nn.Module):
    # 全局平均池化层可通过将池化窗⼝形状设置成输⼊的⾼和宽实现
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()

    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=x.size()[2:])

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm3d(in_channels),
                        nn.ReLU(),
                        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.Dropout(0.5))
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输⼊和输出连结
        return X


def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(),
        nn.Conv3d(in_channels, out_channels, kernel_size=1),
        nn.Dropout(0.5),
        nn.AvgPool3d(kernel_size=2, stride=2))
    return blk

net = nn.Sequential(
    nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm3d(16),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    )

num_channels, growth_rate = 16, 32 # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlock_%d" % i, DB)
    # 上⼀个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加⼊通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net.add_module("BN", nn.BatchNorm3d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", GlobalAvgPool3d())
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 2)))

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in net.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')