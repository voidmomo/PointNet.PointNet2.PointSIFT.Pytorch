# coding:utf-8

import torch

import argparse
import math
import h5py
import numpy as np
import socket
import importlib
import os
import sys

import train
import eval

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'models'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='pointnet',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--dataset', type=str, default='modelnet40', help='the used dataset ')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--train', type=bool, default=False, help='the flag of training or testing')
parser.add_argument('--clean', type=bool, default=False, help='the flag of cleaning the save files or not')
parser.add_argument('--modelnet10', type=bool, default=False, help='Whether to use the modelnet10 in modelnet_normal_resample')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--save_path', type=str, default='/home/guozihao/Workspace/Save/PointNet_plus',
                    help='the path that save the model')


args = parser.parse_args()

if args.train:
    train.start(args)
else:
    eval.start(args)
