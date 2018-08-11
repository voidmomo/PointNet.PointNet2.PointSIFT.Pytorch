# coding:utf-8

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='pointnet',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--dataset', type=str, default='modelnet40', help='the used dataset ')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--train', type=bool, default=False, help='the flag of training or testing')
parser.add_argument('--modelnet10', type=bool, default=False,
                    help='Whether to use the modelnet10 in modelnet_normal_resample')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--save_path', type=str, default='/home/guozihao/Workspace/Save/PointNet',
                    help='the path that save the model')

args = parser.parse_args()
