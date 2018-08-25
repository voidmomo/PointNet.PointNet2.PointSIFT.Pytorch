# coding:utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# ------------------------ modules ------------------------ #
def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq


def fc_bn(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.ReLU()
    )


class Reshape(nn.Module):
    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        size = x.size()
        if self.shape is None:  # 为x增加一个channel维度，并且num_channel=1
            return x.view((size[0], 1) + size[1:])
        return x.contiguous().view(tuple([size[0], ]) + tuple(self.shape))


class Matmul(nn.Module):
    def __init__(self, weights, bias=None):
        super(Matmul, self).__init__()
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        if self.bias is None:
            return torch.matmul(x, self.weights)
        return torch.matmul(x, self.weights) + self.bias


class Input_transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """

    def __init__(self, config, K=3):
        super(Input_transform_net, self).__init__()
        self.num_point = config.num_point
        self.transform_xyz_weights = nn.Parameter(torch.zeros(256, 3 * K))
        self.transform_xyz_bias = nn.Parameter(torch.zeros(3 * K) + torch.FloatTensor(np.eye(3).flatten()))

        self.num_features = 0
        self.net = nn.Sequential(
            Reshape(),
            conv_bn(1, 64, [1, 3]),
            conv_bn(64, 128, [1, 1]),
            conv_bn(128, 1024, [1, 1]),
            nn.MaxPool2d([self.num_point, 1]),
            Reshape([-1, ]),
            fc_bn(1024, 512),
            fc_bn(512, 256),
            Matmul(self.transform_xyz_weights, self.transform_xyz_bias),
            Reshape([3, K])
        )

    def forward(self, x):
        return self.net(x)


class Feature_transform_net(nn.Module):
    """ Feature Transform Net, input is Bx64xN*1
        Return:
            Transformation matrix of size KxK """

    def __init__(self, config, K=64):
        super(Feature_transform_net, self).__init__()
        self.num_point = config.num_point

        self.transform_feat_weights = nn.Parameter(torch.zeros(256, K * K))
        self.transform_feat_bias = nn.Parameter(torch.zeros(K * K) + torch.FloatTensor(np.eye(K).flatten()))

        self.num_features = 0
        self.net = nn.Sequential(
            conv_bn(64, 64, [1, 1]),
            conv_bn(64, 128, [1, 1]),
            conv_bn(128, 1024, [1, 1]),
            nn.MaxPool2d([self.num_point, 1]),
            Reshape([-1, ]),
            fc_bn(1024, 512),
            fc_bn(512, 256),
            Matmul(self.transform_feat_weights, self.transform_feat_bias),
            Reshape([K, K])
        )

    def forward(self, x):
        return self.net(x)


class PointNet_SA_module_basic(nn.Module):
    def __init__(self):
        super(PointNet_SA_module_basic, self).__init__()

    def index_points(self, points, idx):
        """
        Description:
            this function select the specific points from the whole points according to the idx.
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

    def square_distance(self, src, dst):
        """
        Description:
            just the simple Euclidean distance fomula，(x-y)^2,
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def farthest_point_sample(self, xyz, npoint):
        """
        Description:
            first we choose a point from the point set randomly, at the same time,
            see it as a centroid.the calculate the distance of the point and any others,
            and choose the farthest as the second centroid.
            repeat until the number of choosed point has arrived npoint.
        Input:
            xyz: pointcloud data, [B, N, C]
            npoint: number of samples
        Return:
            centroids: the index sampled pointcloud data, [B, npoint, C]
        """
        device = xyz.device
        B, N, C = xyz.shape
        Np = npoint
        centroids = torch.zeros(B, Np, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(Np):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def farthest_point_sample_uniform(self, xyz, npoint):
        """
           Description:
                different with the front function.the function choose the next centroid by
                calculate the distance of one point with other centroids, rather than other point.
                finally, get the max distance.
           Input:
               xyz: pointcloud data, [B, N, C]
               npoint: number of samples
           Return:
               centroids: sampled pointcloud data, [B, npoint, C]
        """
        pass

    def knn(self, xyz, npoint):
        """
           Description:
               first we choose a point from the point set randomly, at the same time,
               see it as a centroid.the calculate the distance of the point and any others,
               and choose the farthest as the second centroid.
               repeat until the number of choosed point has arrived npoint.
           Input:
               xyz: pointcloud data, [B, N, C]
               npoint: number of samples
           Return:
               centroids: sampled pointcloud data, [B, npoint, C]
       """
        pass

    def ball_query(self, radius, nsample, xyz, new_xyz):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, Np, C]
        Return:
            group_idx: grouped points index, [B, Np, Ns]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, Np, _ = new_xyz.shape
        Ns = nsample
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, Np, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :Ns]
        group_first = group_idx[:, :, 0].view(B, Np, 1).repeat([1, 1, Ns])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        """
        Input:
            npoint: the number of points that make the local region.
            radius: the radius of the local region
            nsample: the number of points in a local region
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        B, N, C = xyz.shape
        Np = npoint
        assert isinstance(Np, int)

        new_xyz = self.index_points(xyz, self.farthest_point_sample(xyz, npoint))
        idx = self.ball_query(radius, nsample, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, idx)
        grouped_xyz -= new_xyz.view(B, Np, 1, C)  # the points of each group will be normalized with their centroid
        if points is not None:
            grouped_points = self.index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        """
        Description:
            Equivalent to sample_and_group with npoint=1, radius=np.inf, and the centroid is (0, 0, 0)
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points


class Pointnet_SA_module(PointNet_SA_module_basic):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):

        super(Pointnet_SA_module, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.conv_bns = nn.Sequential()
        in_channel += 3  # +3是因为points 与 xyz concat的原因
        for i, out_channel in enumerate(mlp):
            m = conv_bn(in_channel, out_channel, 1)
            self.conv_bns.add_module(str(i), m)
            in_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: the shape is [B, N, 3]
            points: thes shape is [B, N, D], the data include the feature infomation
        Return:
            new_xyz: the shape is [B, Np, 3]
            new_points: the shape is [B, Np, D']
        """

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # change size to (B, C, Np, Ns), adaptive to conv
        # print("1:", new_points.shape)
        new_points = self.conv_bns(new_points)
        # print("2:", new_points.shape)
        new_points = torch.max(new_points, 3)[0]  # 取一个local region里所有sampled point特征对应位置的最大值。

        new_points = new_points.permute(0, 2, 1).contiguous()
        # print(new_points.shape)
        return new_xyz, new_points


class Pointnet_SA_MSG_module(PointNet_SA_module_basic):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(Pointnet_SA_MSG_module, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        assert len(self.radius_list) == len(self.nsample_list)

        self.in_channel = in_channel
        self.sequentials = nn.ModuleList()
        for sid, mlp in enumerate(mlp_list):
            seq = nn.Sequential()
            self.in_channel = in_channel + 3
            for mid, out_channel in enumerate(mlp):
                m = conv_bn(self.in_channel, out_channel, 1)
                seq.add_module(str(mid), m)
                self.in_channel = out_channel
            self.sequentials.append(seq)

    def forward(self, xyz, points):
        """
        Input:
            xyz: the shape is [B, N, 3]
            points: the shape is [B, N, D]
        Return:
            new_xyz: the shape is [B, Np, 3]
            new_ points: the shape is [B, Np, D']
        """
        (B, N, C), Np = xyz.shape, self.npoint
        new_xyz = self.index_points(xyz, self.farthest_point_sample(xyz, Np))  # B, Np, C
        cat_new_points = []
        for i, radius in enumerate(self.radius_list):
            grouped_idx = self.ball_query(radius, self.nsample_list[i], xyz, new_xyz)  # B, Np, Ns
            grouped_xyz = self.index_points(xyz, grouped_idx)
            grouped_xyz -= new_xyz.view(B, Np, 1, C)  # B, Np, Ns , C
            if points is None:
                grouped_points = grouped_xyz  # B, Np, Ns, C
            else:
                grouped_points = self.index_points(points, grouped_idx)  # B, Np, Ns, D
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # B, Np, Ns, C+D

            grouped_points = grouped_points.permute(0, 3, 2, 1).contiguous()
            new_points = self.sequentials[i](grouped_points)
            new_points = torch.max(new_points, 2)[0]  # B, D', Np
            cat_new_points.append(new_points)
        new_points = torch.cat(cat_new_points, dim=1).permute(0, 2, 1).contiguous()  # B, Np, D'
        return new_xyz, new_points


class PointSIFT_module_basic(nn.Module):
    def __init__(self):
        super(PointSIFT_module_basic, self).__init__()

    def index_points(self, points, idx):
        """
        Description:
            this function select the specific points from the whole points according to the idx.
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

    def pointsift_select_c(self, radius, xyz):
        """
        code by c/c++ logic
        :param radius:
        :param xyz:
        :return:
        """
        Dist = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
        B, N, _ = xyz.shape
        idx = torch.empty(B, N, 8)
        judge_dist = radius ** 2
        temp_dist = torch.ones(B, N, 8) * 1e10
        for b in range(B):
            for n in range(N):
                idx[b, n, :] = n
                x, y, z = xyz[b, n]
                for p in range(N):
                    if p == n: continue
                    tx, ty, tz = xyz[b, p]
                    dist = Dist(x - tx, y - ty, z - tz)
                    if dist > judge_dist: continue
                    _x, _y, _z = tx > x, ty > y, tz > z
                    temp_idx = (_x * 4 + _y * 2 + _z).int()
                    if dist < temp_dist[b, n, temp_idx]:
                        idx[b, n, temp_idx] = p
                        temp_dist[b, n, temp_idx] = dist
        return idx.int()

    def pointsift_select(self, radius, xyz):
        """
        code by python matrix logic
        :param radius:
        :param xyz:
        :return: idx
        """
        dev = xyz.device
        B, N, _ = xyz.shape
        judge_dist = radius ** 2
        idx = torch.arange(N).repeat(8, 1).permute(1, 0).contiguous().repeat(B, 1, 1).to(dev)
        for n in range(N):
            distance = torch.ones(B, N, 8).to(dev) * 1e10
            distance[:, n, :] = judge_dist
            centroid = xyz[:, n, :].view(B, 1, 3).to(dev)
            dist = torch.sum((xyz - centroid) ** 2, -1)  # shape: (B, N)
            subspace_idx = torch.sum((xyz - centroid + 1).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev),
                                     -1)
            for i in range(8):
                mask = (subspace_idx == i) & (dist > 1e-10) & (dist < judge_dist)  # shape: (B, N)
                distance[..., i][mask] = dist[mask]
                idx[:, n, i] = torch.min(distance[..., i], dim=-1)[1]
        return idx

    def pointsift_group(self, radius, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        assert C == 3
        idx = self.pointsift_select(radius, xyz)  # B, N, 8

        grouped_xyz = self.index_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.index_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points, idx

    def pointsift_group_with_idx(self, idx, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        grouped_xyz = self.index_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.index_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points


class PointSIFT_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel):
        super(PointSIFT_module, self).__init__()
        self.radius = radius
        self.conv1 = nn.Sequential(
            conv_bn(3, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

    def forward(self, x):
        pass


class PointSIFT_res_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_res_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3 + extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = nn.Sequential(
            conv_bn(3 + output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2], activation=None)
        )
        if same_dim:
            self.convt = nn.Sequential(
                nn.Conv1d(extra_input_channel, output_channel, 1),
                nn.BatchNorm1d(output_channel),
                nn.ReLU()
            )

    def forward(self, xyz, points):
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        ##print(new_points.shape)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        _, grouped_points = self.pointsift_group_with_idx(idx, xyz, new_points)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        ##print(grouped_points.shape)
        new_points = self.conv2(grouped_points)

        new_points = new_points.squeeze(-1)

        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
            # print(points.shape)
            if self.same_dim:
                points = self.convt(points)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = torch.cat([new_points, points], dim=1)

        new_points = F.relu(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()

        return xyz, new_points


# ------------------------ models ------------------------ #
class PointNet(nn.Module):

    def __init__(self):
        '''
        input: B x N x 3
        output: B x 40
        '''
        super(PointNet, self).__init__()

        from config.config_pointnet import config
        self.num_point = config.num_point
        self.num_classes = config.num_classes
        self.K = config.K

        self.input_transform_net = Input_transform_net(config)
        self.feat_transform_net = Feature_transform_net(config, self.K)
        self.conv1 = conv_bn(1, 64, [1, 3])
        self.conv2 = conv_bn(64, self.K, [1, 1])
        self.conv3 = conv_bn(self.K, 64, [1, 1])
        self.conv4 = conv_bn(64, 128, [1, 1])
        self.conv5 = conv_bn(128, 1024, [1, 1])
        self.fc1 = fc_bn(1024, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # self.I = nn.Parameter(torch.tensor(np.eye(config.K), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.initialize_weights()

    def forward(self, x):

        B = x.size()[0]
        input_transform = self.input_transform_net(x)
        x = torch.matmul(x, input_transform)
        x = x.view((B, 1) + x.size()[1:])
        x = self.conv1(x)
        x = self.conv2(x)

        from config.config_pointnet import config
        feat_transform = self.feat_transform_net(x)
        config.end_point['transform'] = feat_transform

        x = x.squeeze(3).permute(0, 2, 1).contiguous()
        x = torch.matmul(x, feat_transform)
        x = x.permute(0, 2, 1).contiguous()  # 这里用转置感觉有问题，后面再仔细思考一下
        x = x.view(x.size() + (1,))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Npymmetric function: max pooling
        x = F.max_pool2d(x, [self.num_point, 1])
        x = x.view([B, -1])
        x = self.fc1(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.dropout(x, p=0.3)
        x = self.fc3(x)

        return x

    @staticmethod
    def get_loss(input, target):

        """
            input: B*NUM_CLANpNpENp,
            target: B,
        """

        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        assert len(loss.size()) < 2

        # Enforce the transformation as orthogonal matrix
        from config.config_pointnet import config
        transform = config.end_point['transform']  # BxKxK
        mat_diff = torch.matmul(transform, transform.transpose(2, 1).contiguous())

        I = mat_diff.new_tensor(torch.eye(config.K))
        mat_diff.sub_(I)

        mat_diff_loss = torch.sum(mat_diff ** 2) / 2

        reg_weight = 0.001
        return loss.to(mat_diff_loss.device) + mat_diff_loss * reg_weight

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PointNet_plus(nn.Module):
    def __init__(self):
        super(PointNet_plus, self).__init__()
        from config.config_pointnet_plus import config
        self.num_classes = config.num_classes

        # set abstraction network
        self.pointnet_sa_msg_m1 = Pointnet_SA_MSG_module(npoint=512, radius_list=[0.1, 0.2, 0.4],
                                                         nsample_list=[16, 32, 128], in_channel=3 - 3,
                                                         mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.pointnet_sa_msg_m2 = Pointnet_SA_MSG_module(npoint=128, radius_list=[0.2, 0.4, 0.8],
                                                         nsample_list=[32, 64, 128], in_channel=323 - 3,
                                                         mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.pointnet_sa_m3 = Pointnet_SA_module(npoint=None, radius=None, nsample=None,
                                                 in_channel=640, mlp=[256, 512, 1024], group_all=True)

        # fully conntected network
        self.fc1 = fc_bn(1024, 512)
        self.dp1 = nn.Dropout(0.4)
        self.fc2 = fc_bn(512, 256)
        self.dp2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, xyz, points=None):
        """
        Input:
            xyz: is the raw point cloud(B * N * 3)
        Return:
        """
        B = xyz.size()[0]
        # #print('xyz:', xyz.shape)
        l1_xyz, l1_points = self.pointnet_sa_msg_m1(xyz, points)
        # #print('l1_xyz:{}, l1_points:{}'.format(l1_xyz.shape, l1_points.shape))
        l2_xyz, l2_points = self.pointnet_sa_msg_m2(l1_xyz, l1_points)
        # #print('l2_xyz:{}, l2_points:{}'.format(l2_xyz.shape, l2_points.shape))
        l3_xyz, l3_points = self.pointnet_sa_m3(l2_xyz, l2_points)
        # print('l3_xyz:{}, l3_points:{}'.format(l3_xyz.shape, l3_points.shape))
        x = l3_points.view(B, 1024)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x

    # def input_split(self, x):
    #     features = None
    #     if x.size(-1) > 3:
    #         features = x[..., 3:].transpose(1, 2).contiguous()
    #     x = x[..., 0:3].contiguous()
    #     return x, features

    @staticmethod
    def get_loss(input, target):
        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        return loss

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PointSIFT(nn.Module):
    def __init__(self):
        super(PointSIFT, self).__init__()

        from config.config_pointsift import config
        self.num_point = config.num_point
        self.num_classes = config.num_classes

        self.pointsift_res_m1 = PointSIFT_res_module(radius=0.1, output_channel=64, merge='concat')
        self.pointnet_sa_m1 = Pointnet_SA_module(npoint=1024, radius=0.1, nsample=32, in_channel=64, mlp=[64, 128],
                                                 group_all=False)

        self.pointsift_res_m2 = PointSIFT_res_module(radius=0.25, output_channel=128, extra_input_channel=128)
        self.pointnet_sa_m2 = Pointnet_SA_module(npoint=256, radius=0.2, nsample=32, in_channel=128, mlp=[128, 256],
                                                 group_all=False)

        self.pointsift_res_m3_1 = PointSIFT_res_module(radius=0.5, output_channel=256, extra_input_channel=256)
        self.pointsift_res_m3_2 = PointSIFT_res_module(radius=0.5, output_channel=512, extra_input_channel=256,
                                                       same_dim=True)

        self.pointnet_sa_m3 = Pointnet_SA_module(npoint=64, radius=0.4, nsample=32, in_channel=512, mlp=[512, 1024],
                                                 group_all=True)

        self.convt = nn.Sequential(
            nn.Conv1d(512 + 256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # fully conntected network
        self.fc1 = fc_bn(1024, 512)
        self.dp1 = nn.Dropout(0.4)
        self.fc2 = fc_bn(512, 256)
        self.dp2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, xyz, points=None):
        """
        Input:
            xyz: is the raw point cloud(B * N * 3)
        Return:
        """
        B = xyz.size()[0]
        # #print('xyz:', xyz.shape)

        # ---- c1 ---- #
        l1_xyz, l1_points = self.pointsift_res_m1(xyz, points)
        # print('l1_xyz:{}, l1_points:{}'.format(l1_xyz.shape, l1_points.shape))
        c1_xyz, c1_points = self.pointnet_sa_m1(l1_xyz, l1_points)
        # print('c1_xyz:{}, c1_points:{}'.format(c1_xyz.shape, c1_points.shape))

        # ---- c2 ---- #
        l2_xyz, l2_points = self.pointsift_res_m2(c1_xyz, c1_points)
        # print('l2_xyz:{}, l2_points:{}'.format(l2_xyz.shape, l2_points.shape))
        c2_xyz, c2_points = self.pointnet_sa_m2(l2_xyz, l2_points)
        # print('c2_xyz:{}, c2_points:{}'.format(c2_xyz.shape, c2_points.shape))

        # ---- c3 ---- #
        l3_1_xyz, l3_1_points = self.pointsift_res_m3_1(c2_xyz, c2_points)
        # print('l3_1_xyz:{}, l3_1_points:{}'.format(l3_1_xyz.shape, l3_1_points.shape))
        l3_2_xyz, l3_2_points = self.pointsift_res_m3_2(l3_1_xyz, l3_1_points)
        # print('l3_2_xyz:{}, l3_2_points:{}'.format(l3_2_xyz.shape, l3_2_points.shape))

        l3_points = torch.cat([l3_1_points, l3_2_points], dim=-1)
        l3_points = l3_points.permute(0, 2, 1).contiguous()
        l3_points = self.convt(l3_points)
        l3_points = l3_points.permute(0, 2, 1).contiguous()
        l3_xyz = l3_2_xyz
        # print('l3_xyz:{}, l3_points:{}'.format(l3_xyz.shape, l3_points.shape))

        # ---- c4 ---- #
        c4_xyz, c4_points = self.pointnet_sa_m3(l3_xyz, l3_points)
        # print(c4_points.shape)

        x = c4_points.view(B, 1024)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def get_loss(input, target):
        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        return loss

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PointSIFT_Seg(nn.Module):
    def __init__(self):
        super(PointSIFT_Seg, self).__init__()

    def forward(self, x):
        pass

    def get_loss(self):
        pass

    def init_weights(self):
        pass


if __name__ == '__main__':
    # from config.config_pointnet import config
    #
    # net = PointNet()
    # # #print(net)
    # x = torch.Tensor(config.batch_size, 2048, 3)
    # y = net(x)
    # #print(y.size())
    # net = PointNet_plus()
    # xyz = torch.rand(config.batch_size, config.num_point, 3)
    #
    # out = net(xyz)
    # # #print(out.shape)

    m = PointSIFT()
    xyz = torch.rand(5, 1048, 3)
    out = m(xyz, None)
