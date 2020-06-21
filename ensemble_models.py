# 导入相关模块
from torch.utils.data import DataLoader, Dataset
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

import mylib

train_img_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val\\train'
val_img_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val\\val'
label_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val.csv'
test_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\test\\test\\candidate'
submission_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\my.csv'


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         block_config=(6, 12, 24, 16),
                         **kwargs)

    return model


# Create model objects
net1 = generate_model(121,
                      num_classes=2,
                      drop_rate=0.5,
                      growth_rate=32,
                      conv1_t_size=3,
                      conv1_t_stride=2,
                      conv1_t_stride2=2)
net2 = generate_model(121,
                      num_classes=2,
                      drop_rate=0.5,
                      growth_rate=22,
                      conv1_t_size=7,
                      conv1_t_stride=2,
                      conv1_t_stride2=2)
net3 = generate_model(121,
                      num_classes=2,
                      drop_rate=0.5,
                      growth_rate=28,
                      conv1_t_size=7,
                      conv1_t_stride=2,
                      conv1_t_stride2=2)

net1.load_state_dict(torch.load(model53_weights))
net2.load_state_dict(torch.load(model575_weights))
net3.load_state_dict(torch.load(model66_weights))


def ensemble_evaluate_accuracy(data_iter, net1, net2, net3):
    acc_sum, n, l_sum = 0.0, 0, 0.0
    for X, y in data_iter:
        net1.eval()  # 评估模式, 这会关闭dropout
        net2.eval()
        net3.eval()
        y_hat1 = net1(X)
        y_hat2 = net2(X)
        y_hat3 = net3(X)
        y_hat = (y_hat1 + y_hat2 + y_hat3) / 3
        l = criterion(y_hat, y).sum()
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        net1.train()  # 改回训练模式
        net2.train()
        net3.train()
        n += y.shape[0]
        l_sum += l.item()
    return l_sum / n, acc_sum / n


prediction = np.zeros(117)
i = 0
for times in tqdm(range(0, 117)):
    while not os.path.exists(test_path + str(i + 1) + '.npz'):
        i = i + 1
    tmp = np.load(test_path + str(i + 1) + '.npz')
    i = i + 1
    img_voxel = tmp['voxel']
    img_mask = tmp['seg']
    x = torch.from_numpy(img_voxel * img_mask * 0.8 + img_voxel * 0.2)[34:66, 34:66, 34:66] / 255
    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    x = x.float()
    # 预测
    net1.eval()  # 评估模式, 这会关闭dropout
    net2.eval()
    net3.eval()
    y_hat1 = net1(X)
    y_hat2 = net2(X)
    y_hat3 = net3(X)
    y_hat = (y_hat1 + y_hat2 + y_hat3) / 3
    net1.train()  # 改回训练模式
    net2.train()
    net3.train()
    arr = y_hat.detach().numpy()[0]
    prediction[times] = np.exp(arr[1]) / (np.exp(arr[0]) + np.exp(arr[1]))
    net.train()

np.savetxt(submission_path, prediction)
