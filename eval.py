# coding: utf-8
import torch

import os
from dataset import ModelNet40
from torch.utils.data import DataLoader


def choose_dataset(args):
    if args.dataset == 'modelnet40':
        transform = None
        dataset = ModelNet40(args, transform)

    return dataset


def choose_model(args, dataset):
    if args.model == 'pointnet':
        from config.config_pointnet import config

        config.num_point = dataset.num_point
        config.num_classes = dataset.num_classes
        config.classes = dataset.classes
        config.totality = len(dataset)

        from network import PointNet
        model = PointNet()

    return model, config


def start(args):
    dataset = choose_dataset(args)

    model, config = choose_model(args, dataset)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    saved_models = os.listdir(args.save_path)
    for sm in saved_models:
        saved_model = os.path.join(args.save_path, sm)
        model.load_state_dict(torch.load(saved_model))
        if args.cuda:
            model.cuda()
        correct = torch.tensor(0)
        for i, data in enumerate(dataloader):
            point_cloud, label = data
            if args.cuda:
                point_cloud, label = point_cloud.cuda(), label.cuda()
            pred = model(point_cloud)
            result = pred.max(1)[1]
            correct += result.eq(label).cpu().sum()
        print('{}, total:{}, correct:{}, accuracy: {} '.format(sm.strip('.pth'), config.totality, correct.item(),
                                                               correct.item() / float(config.totality)))
