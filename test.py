# coding:utf-8

import numpy as np
import torch


def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    device = xyz.device
    B, N, C = xyz.shape
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :K]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


import torch.nn as nn
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp = nn.Linear(50, 10)

    def forward(self, x):
        x = self.mlp(x)
        print("x.shape:", x.shape)
        return x


#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = Net()
# net.to(device)
# input = torch.rand(1024000, 50).cuda()
# last = time.time()
# output = net(input)
# print(output)
# print("single gpu:", time.time() - last)
#
# net = nn.DataParallel(net)
# net.cuda()
# last = time.time()
# output = net(input)
# print(output)
# print("multi gpus:", time.time() - last)

class A:

    def __init__(self):
        self.a = 10

    @staticmethod
    def call(s):
        print('i am call {}'.format(s))


#

# a = torch.tensor([[0.38, -0.32], [-0.45, 0.54]])
#
# print(a)
# b = a.int()
# print(b)
# c = (a + 1).int()
# print(c)
# print(a * torch.tensor([2., 1.]))
import tensorflow as tf

tf.train.NewCheckpointReader()