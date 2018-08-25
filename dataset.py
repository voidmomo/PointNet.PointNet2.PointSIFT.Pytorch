# coding:utf-8

import os
import sys
import h5py
import os.path as osp
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from skimage import transform, io
import pickle


class ModelNet40(Dataset):
    def __init__(self, args, transform=None):
        self.root_dir = '/unsullied/sharefs/_research_detection/GeneralDetection/ModelNet40/ModelNet40'

        self.train_files = self.get_files(osp.join(self.root_dir, 'train_files.txt'))
        self.eval_files = self.get_files(osp.join(self.root_dir, 'test_files.txt'))

        self.classes = self.get_files(osp.join(self.root_dir, 'shape_names.txt'))
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

        self.train = args.train


        if self.train:
            self.num_point = 1024
            self.train_data, self.train_label = self.get_data_and_label(self.train_files)
        else:
            self.num_point = 1024
            self.eval_data, self.eval_label = self.get_data_and_label(self.eval_files)

        self.transform = transform

    def __getitem__(self, item):

        if self.train:
            train_data, train_label = self.train_data[item], self.train_label[item]
            if self.transform and np.random.random() > 0.5:
                train_data = self.transform(train_data)
            return torch.from_numpy(train_data), torch.tensor(train_label, dtype=torch.long)
        else:
            eval_data, eval_label = self.eval_data[item], self.eval_label[item]
            if self.transform:
                eval_data = self.transform(eval_data)
            return torch.from_numpy(eval_data), torch.tensor(eval_label, dtype=torch.long)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.eval_data)

    def get_files(self, fp):
        print(fp)
        with open(fp, 'r') as f:
            files = f.readlines()
        return [f.rstrip() for f in files]

    def get_data_and_label(self, files):
        all_data, all_label = [], []
        for fn in files:
            print('---------' + str(fn) + '---------')
            current_data, current_label = self.load_h5(osp.join(self.root_dir, fn))
            current_data = current_data[:, 0:self.num_point, :]
            current_label = np.squeeze(current_label)
            all_data.append(current_data)
            all_label.append(current_label)

        return np.concatenate(all_data), np.concatenate(all_label)

    def load_h5(self, fp):
        f = h5py.File(fp)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def load_h5_data_label_seg(self, fp):
        f = h5py.File(fp)
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        return (data, label, seg)


class ModelNet_Normal_Resampled(Dataset):
    '''
        ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
    '''

    def __init__(self, args, transform):
        self.root_dir = '/unsullied/sharefs/_research_detection/GeneralDetection/ModelNet40/modelnet40_normal_resampled'

        if args.modelnet10:
            self.train_files = self.get_files(osp.join(self.root_dir, 'modelnet10_train.txt'))
            self.eval_files = self.get_files(osp.join(self.root_dir, 'modelnet10_test.txt'))
            self.classes = self.get_files(osp.join(self.root_dir, 'modelnet10_shape_names.txt'))
        else:
            self.train_files = self.get_files(osp.join(self.root_dir, 'modelnet40_train.txt'))
            self.eval_files = self.get_files(osp.join(self.root_dir, 'modelnet40_test.txt'))
            self.classes = self.get_files(osp.join(self.root_dir, 'shape_names.txt'))
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

        self.num_point = 1024
        self.num_channel = 3
        if args.normal:
            self.num_channel = 6

        self.train = args.train
        if self.train:
            self.train_data, self.train_label = self.get_data_and_label(self.train_files, self.train)
        else:
            self.eval_data, self.eval_label = self.get_data_and_label(self.eval_files, self.train)

        self.transform = transform

    def __getitem__(self, item):
        if self.train:
            train_data, train_label = self.train_data[item], self.train_label[item]
            if self.transform:
                train_data = self.transform(train_data)
            return torch.from_numpy(train_data), torch.tensor(train_label, dtype=torch.long)
        else:
            eval_data, eval_label = self.eval_data[item], self.eval_label[item]
            if self.transform:
                eval_data = self.transform(eval_data)
            return torch.from_numpy(eval_data), torch.tensor(eval_label, dtype=torch.long)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.eval_data)

    def get_files(self, fp):
        print(fp)
        with open(fp, 'r') as f:
            files = f.readlines()
        return [f.rstrip() for f in files]


class KITTI(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class ApolloScape(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class ScannetDataset():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % split)
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        smpmin = np.maximum(coordmax - [1.5, 1.5, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax - smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2] - coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None
        mask = None
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(
                vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice, :]

        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetWholeScene(object):
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % split)
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
        nsubvolume_x *= 2
        nsubvolume_y *= 2
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 0.75, j * 0.75, 0]
                curmax = coordmin + [i * 0.75 + 1.5, j * 0.75 + 1.5, coordmax[2] - coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                if sum(mask) < 2000:
                    continue
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N

                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    m = ModelNet40()
