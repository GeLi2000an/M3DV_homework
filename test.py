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

train_img_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val\\train'
val_img_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val\\val'
label_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\train_val.csv'
test_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\test\\test\\candidate'
submission_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\my.csv'

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

batch_size = 64

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

train_dataset = myDataset(train_img_path, label_path, transform=normalize)
val_dataset = myDataset(val_img_path, label_path, transform=normalize)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


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

lr = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate_accuracy(data_iter, net):
    acc_sum, n, l_sum = 0.0, 0, 0.0
    for X, y in data_iter:
        net.eval() # 评估模式, 这会关闭dropout
        y_hat = net(X)
        l = criterion(y_hat, y).sum()
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        net.train() # 改回训练模式
        n += y.shape[0]
        l_sum += l.item()
    return l_sum / n, acc_sum / n


import copy

weights_path = 'C:\\Users\\dell\\Desktop\\sjtu-ee228-2020\\densenet25.pth'


def train(net, optimizer, num_epochs, best_val_acc, best_val_loss):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for images, labels in train_loader:
            """
            mixed_x, y_a, y_b, lam = mixup_data(images, labels)
            labels_hat = net(mixed_x)
            l = mixup_criterion(criterion, labels_hat, y_a, y_b, lam).float().sum()
            """
            labels_hat = net(images)
            l = criterion(labels_hat, labels).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (labels_hat.argmax(dim=1) == labels).sum().item()
            n += labels.shape[0]

        # evaluate
        val_loss, val_acc = evaluate_accuracy(val_loader, net)
        print('epoch %d, loss %.4f, train acc %.3f, val loss %.3f, val acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, val_loss, val_acc))
        '''
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            net_copy = copy.deepcopy(net.state_dict())
        '''
    # net.load_state_dict(net_copy)
    # print('best_val_acc %.3f, epoch %d' % (best_val_acc, epoch+1))
    return net


net = train(net, optimizer, num_epochs=1, best_val_acc=0.6, best_val_loss=0.5)

prediction = np.zeros(117)
i = 0
for times in tqdm(range(0, 117)):
    while not os.path.exists(test_path + str(i+1) + '.npz'):
        i = i + 1
    tmp = np.load(test_path + str(i+1) + '.npz')
    i = i + 1
    img_voxel = tmp['voxel']
    img_mask = tmp['seg']
    x = torch.from_numpy(img_voxel * img_mask * 0.8 + img_voxel * 0.2)[34:66, 34:66, 34:66] / 255
    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    x = x.float()
    # 预测
    net.eval()
    y_hat = net(x)
    arr = y_hat.detach().numpy()[0]
    prediction[times] = np.exp(arr[1])/(np.exp(arr[0]) + np.exp(arr[1]))
    net.train()

np.savetxt(submission_path, prediction)
