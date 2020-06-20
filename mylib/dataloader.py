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

def csvDictReader(path):
    with open(path) as rf:
        reader = csv.reader(rf)
        items = list(reader)
    dict = {}
    for line in items:
        dict[line[0] + '.npz'] = int(line[1])
    return dict


class myDataset(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.img = os.listdir(img_path)  # img是一个list
        self.labelDict = csvDictReader(label_path)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img_index = self.img[index]
        img_path = os.path.join(self.img_path, img_index)
        img_voxel = np.load(img_path)['voxel']
        img_mask = np.load(img_path)['seg']
        final_img = img_voxel * img_mask * 0.8 + img_voxel * 0.2;
        final_img = final_img[34:66, 34:66, 34: 66]
        final_img = torch.from_numpy(final_img) / 255
        final_img = torch.unsqueeze(final_img, 0)
        label = self.labelDict[img_index]

        return final_img.float(), label