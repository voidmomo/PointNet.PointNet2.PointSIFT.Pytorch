# coding: utf-8
import torch

from dataset import ModelNet40, ModelNet_Normal_Resampled
from torch.utils.data import DataLoader
from utils import RotatePointCloud, RotatePointCloud_Normal, JitterPointCloud
import torchvision.transforms as transforms
import torch.optim as optim
import math


def choose_dataset(args):
    if args.dataset == 'modelnet40':
        transform = transforms.Compose([RotatePointCloud(), JitterPointCloud()])
        dataset = ModelNet40(args, transform)

    elif args.dataset == 'modelnet_normal_resampled':
        if args.normal:
            transform = transforms.Compose([RotatePointCloud_Normal, JitterPointCloud])
        else:
            transform = transforms.Compose([RotatePointCloud(), JitterPointCloud()])
        dataset = ModelNet_Normal_Resampled(args, transform)

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
        from config.config_pointsift import config

        from network import PointNet_plus
        model = PointNet_plus()

    return model, config


def lr_exponential_decay(optimizer, global_step, decay_rate, decay_step):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * math.pow(decay_rate, global_step / float(decay_step))
        param_group['lr'] = max(param_group['lr'], 1e-5)


def start(args):
    dataset = choose_dataset(args)

    model, config = choose_model(args, dataset)

    if args.cuda:
        model.cuda()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(args.max_epoch + 1):
        for i, data in enumerate(dataloader):
            point_cloud, label = data
            if args.cuda:
                point_cloud, label = point_cloud.cuda(), label.cuda()
            optimizer.zero_grad()
            pred = model(point_cloud)
            train_loss = model.get_loss(pred, label)
            train_loss.backward()
            optimizer.step()
            lr_exponential_decay(optimizer, epoch * (config.totality / config.batch_size) + i, config.decay_rate,
                                 config.decay_step)
            if i % 16 == 0:
                result = pred.max(1)[1]
                correct = result.eq(label).cpu().sum()
                print('[%d: %d/%d] lr: %f train loss: %f accuracy: %f ' % (
                    epoch, i * config.batch_size, config.totality, optimizer.param_groups[0]['lr'], train_loss.item(),
                    correct.item() / float(config.batch_size)))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '%s/cls_model_%d.pth' % (args.save_path, epoch))
