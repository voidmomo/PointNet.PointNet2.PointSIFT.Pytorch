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


# xyz = torch.rand(3, 4, 6)
# npoint = 2
#
# print(xyz)
# res = farthest_point_sample(xyz, npoint)
# print(res)
import torch.nn as nn
# print(("%03d"%4))
# import torch.nn.functional as F
#
# # a = torch.rand(3, 4)
# # print(a)
# # print(torch.log(F.softmax(a, 1)))
# # print(F.log_softmax(a, 1))
#
# # F.nll_loss()
# input = torch.randn(3, 5, requires_grad=True)
# # each element in target has to have 0 <= value < C
# target = torch.tensor([1, 0, 4])
# output = F.nll_loss(F.log_softmax(input, dim=1), target)
# output2 = F.cross_entropy(input, target)
# print(output.detach())
# print(output2.detach())

# print(res)
# res = ball_query(0.2,1, xyz, res)
# print(res.shape,res)
# res = index_points(xyz, res)
# print(res.shape, res)
#

import torch.nn as nn

# net = nn.Sequential()
# net.append(nn.Linear(25, 24))
# x = torch.rand(25)
# y = net(x)
# print(y.shape)

# model = nn.Sequential(
#     nn.Conv2d(1, 20, 5),
#     nn.ReLU(),
#     nn.Conv2d(20, 64, 5),
#     nn.ReLU()
# )
# for idx, module in enumerate(model.modules()):
#     print(idx)

# a = torch.rand(32, 512)
# b = [a, a, a]
# c = torch.cat(b, dim=1)
# print(c.shape)

# import torch.nn.functional as F
# from tqdm import tqdm
# from tqdm import trange
# from time import sleep
#
# a = [['b'] for i in range(10)]
#
# with tqdm(a) as pbar:
#     for i, s in enumerate(pbar):
#         pbar.update(i)
#         sleep(1)
class A:
    ab = 20
    def __init__(self):
        self.ab = A.ab+8

    @classmethod
    def b(cls):
        print(cls.ab)

    @staticmethod
    def a():
        print("this is a")


A.b()
a = A()
print(a.ab)