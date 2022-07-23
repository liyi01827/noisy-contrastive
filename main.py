import argparse
import time
import math
from os import path, makedirs
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import copy
import torch.nn as nn
from simsiam.model_factory import SimSiam


from loader import CIFAR10N, CIFAR100N
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint, accuracy, load_checkpoint, ThreeCropsTransform


parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default='~/data', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default='./save', type=str, help='path to experiment directory')
parser.add_argument('--dataset', default='cifar10', type=str, help='path to dataset', choices=["cifar10", "cifar100"])
parser.add_argument('--noise_type', default='sym', type=str, help='noise type: sym or asym', choices=["sym", "asym"])
parser.add_argument('--r', type=float, default=0.8, help='noise level')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)

parser.add_argument('--arch', default='resnet18', help='model name is used for training')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=550, help='number of training epochs')

parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--m', type=float, default=0.99, help='moving average of probbility outputs')
parser.add_argument('--tau', type=float, default=0.8, help='contrastive threshold (tau)')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--lamb', default=50.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--type', default='ce', type=str, help='ce or gce loss', choices=["ce", "gce"])
parser.add_argument('--beta', default=0.6, type=float, help='gce parameter')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

args = parser.parse_args()
import random
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.dataset == 'cifar10':
    args.nb_classes = 10
elif args.dataset == 'cifar100':
    args.nb_classes = 100


class GCE_loss(nn.Module):
    def __init__(self, q=0.8):
        super(GCE_loss, self).__init__()
        self.q = q

    def forward(self, outputs, targets):
        targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = F.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss


if args.type == 'ce':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = GCE_loss(args.beta)


def set_model(args):
    model = SimSiam(args.m, args)
    model.cuda()
    return model



def set_loader(args):
    if args.dataset == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        train_set = CIFAR10N(root=args.data_root,
                             transform=ThreeCropsTransform(train_transforms, train_cls_transformcon),
                             noise_type=args.noise_type,
                             r=args.r)

        val_set = copy.deepcopy(train_set)
        val_set.transform = test_transform
        val_loader = DataLoader(dataset=val_set,
                                batch_size=128,
                                shuffle=False,
                                num_workers=args.num_workers)

        test_data = datasets.CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)
    elif args.dataset == 'cifar100':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])

        train_set = CIFAR100N(root=args.data_root,
                              transform=ThreeCropsTransform(train_transforms, train_cls_transformcon),
                              noise_type=args.noise_type,
                              r=args.r)

        val_set = copy.deepcopy(train_set)
        val_set.transform = test_transform
        val_loader = DataLoader(dataset=val_set,
                                batch_size=128,
                                shuffle=False,
                                num_workers=args.num_workers)

        test_data = datasets.CIFAR100(root=args.data_root, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    return train_loader, test_loader



def train(train_loader, model, criterion, optimizer, epoch,  args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, targets, _, index) in enumerate(train_loader):
        bsz = targets.size(0)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        p1, z2, outputs = model(images[0], images[1], images[2])


        # avoid clapse and gradient explosion
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)

        contrast_1 = torch.matmul(p1, z2.t())  # B X B


        # <q,z> + log(1-<q,z>)
        contrast_1 = -contrast_1*torch.zeros(bsz, bsz).fill_diagonal_(1).cuda() + ((1-contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda()
        contrast_logits = 2 + contrast_1


        soft_targets = torch.softmax(outputs, dim=1)
        contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
        contrast_mask.fill_diagonal_(1)
        pos_mask = (contrast_mask >= args.tau).float()
        contrast_mask = contrast_mask * pos_mask
        contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)
        loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)

        loss_ce = criterion(outputs, targets)


        loss = args.lamb*loss_ctr + loss_ce

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    return losses.avg




def validation(test_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    acc = AverageMeter('Loss', ':.4e')

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                # targets = targets.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs = model.forward_test(images)
            acc2 = accuracy(outputs, targets, topk=(1,))

            # measure elapsed time
            acc.update(acc2[0].item(), images[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    return acc.avg


def main():
    print(vars(args))

    train_loader, test_loader = set_loader(args)

    model = set_model(args)


    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

    start_epoch = 0


    # routine
    best_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_optim = epoch

        adjust_learning_rate(optimizer, epoch_optim, args)
        print("Training...")

        # train for one epoch
        time0 = time.time()
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        print("Train \tEpoch:{}/{}\ttime: {}\tLoss: {}".format(epoch, args.epochs, time.time()-time0, train_loss))

        time0 = time.time()
        val_top1_acc = validation(test_loader, model, epoch, args)
        print("Test\tEpoch:{}/{}\t time: {}\tAcc: {}".format(epoch, args.epochs, time.time()-time0, val_top1_acc))
        best_acc = max(best_acc, val_top1_acc)

        # scheduler.step()


    with open('log.txt', 'a') as f:
        if args.type == 'ce':
            f.write('dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: ce \t seed: {} \t best_acc: {}\tlast_acc: {}\n'.format(args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.seed, best_acc, val_top1_acc))
        elif args.type == 'gce':
            f.write('dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {} \t best_acc: {}\tlast_acc: {}\n'.format(args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed, best_acc, val_top1_acc))


if __name__ == '__main__':
    main()



