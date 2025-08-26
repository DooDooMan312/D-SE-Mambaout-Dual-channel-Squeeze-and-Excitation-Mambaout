# -*- coding: utf-8 -*-   
# @File  : my_mamba.py         
# @Author : Liu_Liuliu
# @Date  :  2025/03/16

import torch.nn as nn
import random

import argparse

import torch

import timm

assert timm.__version__ == "0.6.11"  # version check

from mambaout import MambaOut
from model_utils.data import getDataLoader
from model_utils.train_v2 import train_VAL
from torch.amp import autocast, GradScaler

import os


def init_mamba_class(pretrained=False, **kwargs):  # 初始化 Mamba
    #todo
    model = MambaOut(
        depths=[3, 3, 9, 3],
        dims=[48, 64, 192, 288],
        **kwargs
        ).to(device)

    # model.default_cfg = default_cfgs['mambaout_base']

    # # 加载预训练权重
    # if pretrained:
    #     url = model.default_cfg.get('url')
    #     if not url:
    #         raise ValueError("Pretrained model URL is missing in default configuration.")
    #     try:
    #         state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, check_hash=True)
    #         model.load_state_dict(state_dict)
    #         print("✅ Loaded pretrained weights.")
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load pretrained weights: {e}")

    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    print("Model weights initialized.")
    return model



##################################################################
# 1是<7.05 2<7.15 3其他

savePath = r'model_save/DASFine'

##########################################
dataset_paths = {
    'FHR':{
        'train': {
            'class1': r'dataset_recurrent_v2/FHR/extra_acid/train',
            'class2': r'dataset_recurrent_v2/FHR/acid/train',
            'class3': r'dataset_recurrent_v2/FHR/fine/train',
        },
        'val': {
            'class1': r'dataset_recurrent_v2/FHR/extra_acid/val',
            'class2': r'dataset_recurrent_v2/FHR/acid/val',
            'class3': r'dataset_recurrent_v2/FHR/fine/val',
        },
        'test': {
            'class1': r'dataset_recurrent_v2/FHR/extra_acid/test',
            'class2': r'dataset_recurrent_v2/FHR/acid/test',
            'class3': r'dataset_recurrent_v2/FHR/fine/test',
        }
},
    'UC': {
        'train': {
            'class1': r'dataset_recurrent_v2/UC/extra_acid/train',
            'class2': r'dataset_recurrent_v2/UC/acid/train',
            'class3': r'dataset_recurrent_v2/UC/fine/train',
        },
        'val': {
            'class1': r'dataset_recurrent_v2/UC/extra_acid/val',
            'class2': r'dataset_recurrent_v2/UC/acid/val',
            'class3': r'dataset_recurrent_v2/UC/fine/val',
        },
        'test': {
            'class1': r'dataset_recurrent_v2/UC/extra_acid/test',
            'class2': r'dataset_recurrent_v2/UC/acid/test',
            'class3': r'dataset_recurrent_v2/UC/fine/test',
        }
    }
}
##########################################
# class1Train = r'dataset/GAF_0/train'
# class2Train = r'dataset/GAF_1/train'
# class3Train = r'dataset/GAF_2/train'
# class1Val = r'dataset/GAF_0/val'
# class2Val = r'dataset/GAF_1/val'
# class3Val = r'dataset/GAF_2/val'
# class1Test = r'dataset/GAF_0/test'
# class2Test = r'dataset/GAF_1/test'
# class3Test = r'dataset/GAF_2/test'
###


# 读数据这里按照自己的写法就行 。
#################################################################
random.seed(1)
learning_rate = 1e-5

# Assuming model is already defined and set to device
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification with class indices
# 之前使用soft，效果很差，现在model_head也没有用softmax


scaler = GradScaler()  # Mixed precision training

batch_size = 1
epoch = 30
# w = 0.00001
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##################################################################

train_FHR = dataset_paths['FHR']['train']
train_UC = dataset_paths['UC']['train']

val_FHR = dataset_paths['FHR']['val']
val_UC = dataset_paths['UC']['val']

test_FHR = dataset_paths['FHR']['test']
test_UC = dataset_paths['UC']['test']


trainloader_FHR = getDataLoader(train_FHR['class1'], train_FHR['class2'], train_FHR['class3'], batchSize=batch_size)  # Dataset12（12为内存地址）: (batch_size, sample, X, Y)
trainloader_UC = getDataLoader(train_UC['class1'], train_UC['class2'], train_UC['class3'], batchSize=batch_size)  # Dataset12: (batch_size, sample, X, Y)

valloader_FHR = getDataLoader(val_FHR['class1'], val_FHR['class2'], val_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
valloader_UC = getDataLoader(val_UC['class1'], val_UC['class2'], val_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)

testloader_FHR = getDataLoader(test_FHR['class1'], test_FHR['class2'], test_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
testloader_UC = getDataLoader(test_UC['class1'], test_UC['class2'], test_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)



if __name__ == '__main__':
    model = init_mamba_class(pretrained=False)
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # Add weight decay

    train_VAL(model, trainloader_FHR, trainloader_UC, valloader_FHR, valloader_UC, testloader_FHR, testloader_FHR,
              optimizer, loss_fn, batch_size,
              num_epoch=epoch, device=device, save_=savePath)

    # modelpath1 = savePath+'max'
    # model1 = torch.load(modelpath1)

    # test(model1, test_set=test_dataset)

    # modelpath2 = savePath+'final'

    # model2 = torch.load(modelpath2)
    # test(model2, test_set=test_dataset)

