# coding: utf-8
import torch

import os
from dataset import ModelNet40
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

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

    elif args.model == 'pointnet_plus':
        from config.config_pointnet_plus import config

        config.num_classes = dataset.num_classes
        config.num_point = dataset.num_point
        config.classes = dataset.classes
        config.totality = len(dataset)

        from network import PointNet_plus
        model = PointNet_plus()
    else:
        raise Exception("The choosed model doesn't exist!")

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
        model.eval()
        correct = torch.tensor(0)
        with tqdm(dataloader) as pbar:
            for i, data in enumerate(pbar):
                point_cloud, label = data
                if args.cuda:
                    point_cloud, label = point_cloud.cuda(), label.cuda()
                pred = model(point_cloud)
                result = F.softmax(pred, dim=1).max(1)[1]
                correct += result.eq(label).cpu().sum().item()
                pbar.update(i)

        print('{}, total:{}, correct:{}, accuracy: {} '.format(sm.strip('.pth'), config.totality, correct,
                                                               float(correct) / config.totality))
