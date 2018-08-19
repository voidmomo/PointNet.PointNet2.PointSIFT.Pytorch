# coding: utf-8
import torch

from dataset import ModelNet40, ModelNet_Normal_Resampled
from torch.utils.data import DataLoader
import utils
from utils import RotatePointCloud, RotatePointCloud_Normal, JitterPointCloud
import torchvision.transforms as transforms
import torch.optim as optim

import math
import os


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
        from network import PointNet

        config.num_point = dataset.num_point
        config.num_classes = dataset.num_classes
        config.classes = dataset.classes
        config.totality = len(dataset)

        if torch.cuda.device_count() > 1:
            config.batch_size *= torch.cuda.device_count()
        return PointNet(), config, PointNet

    elif args.model == 'pointnet_plus':
        from config.config_pointnet_plus import config

        config.num_classes = dataset.num_classes
        config.num_point = dataset.num_point
        config.classes = dataset.classes
        config.totality = len(dataset)

        if torch.cuda.device_count() > 1:
            config.batch_size *= torch.cuda.device_count()

        from network import PointNet_plus
        return PointNet_plus(), config, PointNet_plus
    else:
        raise Exception("The choosed model doesn't exist!")
        exit()


def lr_exponential_decay(optimizer, global_step, decay_rate, decay_step):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * math.pow(decay_rate, global_step / float(decay_step))
        param_group['lr'] = max(param_group['lr'], 1e-5)


def start(args):
    dataset = choose_dataset(args)

    model, config, Net = choose_model(args, dataset)

    if torch.cuda.device_count() > 1:
        print("we will use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    if args.cuda and torch.cuda.is_available():
        model.cuda()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    utils.check_filepath(args.save_path, args.clean)
    print("save path:{}".format(args.save_path))
    saved_models = os.listdir(args.save_path)
    last_epoch = 1

    if len(saved_models) > 0:
        saved_epochs = [int(model.rstrip('.pth')[-1]) for model in saved_models]
        last_epoch = str(max(saved_epochs))
        saved_model = os.path.join(args.save_path, 'cls_model_{}.pth'.format(last_epoch))
        print("load weight: {}".format(saved_model))
        model.load_state_dict(torch.load(saved_model))

    for epoch in range(last_epoch, args.max_epoch + 1):

        total_train_loss = 0.
        total_correct = 0
        model.train()

        for i, data in enumerate(dataloader):
            point_cloud, label = data
            if args.cuda:
                point_cloud, label = point_cloud.cuda(), label.cuda()
            optimizer.zero_grad()
            pred = model(point_cloud)
            train_loss = Net.get_loss(pred, label)
            train_loss.backward()
            optimizer.step()
            lr_exponential_decay(optimizer, epoch * (config.totality / config.batch_size) + i, config.decay_rate,
                                 config.decay_step)
            result = pred.max(1)[1]
            correct = result.eq(label).cpu().sum()
            total_correct += correct.item()
            total_train_loss += train_loss.item()
            if i % 16 == 0:
                print('[%d: %d/%d] lr: %f train loss: %f accuracy: %f ' % (
                    epoch, i * config.batch_size, config.totality, optimizer.param_groups[0]['lr'], train_loss.item(),
                    correct.item() / float(config.batch_size)))
        print('[- %d -] train loss: %f accuracy: %.4f ' % (
            epoch, total_train_loss / len(dataloader), float(total_correct) / config.totality))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '%s/cls_model_%03d.pth' % (args.save_path, epoch))
