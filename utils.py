# coding: utf-8


import torch
import numpy as np
from torch.autograd import Function
import os


# --------------------- Transforms --------------------- #

class RotatePointCloud(object):
    """ Randomly rotate the point clouds to augument the dataset
           rotation is per shape based along up direction
           Input:
             BxNx3 array, original batch of point clouds
           Return:
             BxNx3 array, rotated batch of point clouds
    """

    def __init__(self, rotation_angle=None):
        self.rotation_angle = rotation_angle

    def __call__(self, batch_data):
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            if self.rotation_angle is None:
                rotation_angle = np.random.uniform() * 2 * np.pi
            else:
                rotation_angle = self.rotation_angle
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data


class RotatePointCloud_Normal(object):
    """ Randomly rotate XYZ, normal point cloud.
         Input:
             batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
         Output:
             B,N,6, rotated XYZ, normal point cloud
    """

    def __init__(self, rotation_angle=None):
        self.rotation_angle = rotation_angle

    def __call__(self, batch_data_normal):
        for k in range(batch_data_normal.shape[0]):
            if self.rotation_angle is None:
                rotation_angle = np.random.uniform() * 2 * np.pi
            else:
                rotation_angle = self.rotation_angle
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data_normal[k, :, 0:3]
            shape_normal = batch_data_normal[k, :, 3:6]
            batch_data_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
            batch_data_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
        return batch_data_normal


class RotatePerturbationPointCloud_Normal(object):
    def __init__(self, sigma=0.06, clip=0.18):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, batch_data_normal):
        rotated_data = np.zeros(batch_data_normal.shape, dtype=np.float32)
        for k in range(batch_data_normal.shape[0]):
            rotation_angle = np.clip(self.sigma * np.random.randn(3), -self.clip, self.clip)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(rotation_angle[0]), -np.sin(rotation_angle[0])],
                           [0, np.sin(rotation_angle[0]), np.cos(rotation_angle[0])]])
            Ry = np.array([[np.cos(rotation_angle[1]), 0, np.sin(rotation_angle[1])],
                           [0, 1, 0],
                           [-np.sin(rotation_angle[1]), 0, np.cos(rotation_angle[1])]])
            Rz = np.array([[np.cos(rotation_angle[2]), -np.sin(rotation_angle[2]), 0],
                           [np.sin(rotation_angle[2]), np.cos(rotation_angle[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = batch_data_normal[k, :, 0:3]
            shape_normal = batch_data_normal[k, :, 3:6]
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
            rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
        return rotated_data


class JitterPointCloud(object):
    """ Randomly jitter points. jittering is per point.
            Input:
              BxNxC array, original batch of point clouds
            Return:
              BxNxC array, jittered batch of point clouds
        """

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, batch_data):
        N, C = batch_data.shape
        assert (self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip).astype(np.float32)
        jittered_data += batch_data
        return jittered_data


# --------------------- Functions --------------------- #

def check_filepath(fp, clean=False):
    if not os.path.exists(fp):
        os.makedirs(fp)
    elif clean:
        ls = os.listdir(fp)
        for l in ls:
            f = os.path.join(fp, l)
            if os.path.isdir(f):
                os.removedirs()
                check_filepath(l, clean=True)
            else:
                os.remove(f)


