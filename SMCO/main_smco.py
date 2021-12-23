import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
import random
from BaseEncoder import ConvMixer
from SmCo import SmCoModel
import torch.nn.functional as F
from utils.MeterUtils import *
import argparse
from utils.my_utils import *
from utils import loader
import time
import shutil
import math

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parse = argparse.ArgumentParser(description="this is the test")
parse.add_argument("--batch-size", default=32, type=int,
                   help="batch size")
parse.add_argument("--epochs", default=100, type=int,
                   help="the epoch")
parse.add_argument("--seed", default=1,
                   help="the random seed")
parse.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parse.add_argument("--k-queue", default=128, type=int,
                   help="the length of the queue")
parse.add_argument("--input-dim", default=3, type=int,
                   help="the moco input image channel dimension defalut is 3")
parse.add_argument("--m", default=0.999, type=float,
                   help="the momentum")
parse.add_argument("--t", default=0.07, type=float,
                   help="the tempture")
parse.add_argument("--output-dim", default=10, type=int,
                   help="the encoder output dimention")
parse.add_argument("--lr", default=0.0025, type=float,
                   help="learning rate")
parse.add_argument("--momentum", default=0.9, type=float,
                   help="the momentum of the SGD")
parse.add_argument("--weight-decay", default=1e-4, type=float,
                   help="the SGD weight decay")
parse.add_argument("--model-path", default="./model",
                   help="the model path")
parse.add_argument("--arch", default='resnet50', choices=model_names,
                   help="the choice of the backbone")
# parse.add_argument("--resume", default="", type=str,
#                    help="path to the latest checkpoint")
parse.add_argument("--resume", default="./model/checkpoint_0000.pth.tar", type=str,
                   help="path to the latest checkpoint")
parse.add_argument("--start-epoch", default=0, type=int,
                   help="manual epoch number only useful on restart")
parse.add_argument("--aug-plus", default=1, type=int, choices=[0, 1, 2],
                   help="aug-plus 0 for MoCo V1; 1 for MoCo V2; 2 for MoCo DK")
parse.add_argument("--num-workers", default=1, type=int,
                   help="the thread number")
parse.add_argument('--schedule', default=[120, 160], type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parse.add_argument('--cos', default=True,
                   help="judge if the cos action is True")
parse.add_argument("--mlp", default=False,
                   help="if need the mlp to do the lincl")


def train(_model, _opt, _epoch, _train_loader, args):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    batch_time = AverageMeter("BatchTime", ":6.4f")
    epoch_time = AverageMeter("EpochTime", ":6.4f")
    progress = ProgressMeter(
        len(_train_loader),
        [losses, top1, top5, batch_time, epoch_time],
        prefix="Epoch:[{}]".format(_epoch)
    )

    _model.train()
    start = time.time()# 用来记录epoch time
    end = time.time()# 用来记录batch_time
    for index, (images, label) in enumerate(_train_loader):
        images[0], images[1] = images[0].to(device=args.device), images[1].to(device=args.device)
        _opt.zero_grad()
        output, target = _model(images[0], images[1])
        loss = F.cross_entropy(output, target).cuda(0)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size()[0])
        top1.update(acc1[0].item(), images[0].size()[0])
        top5.update(acc5[0].item(), images[0].size()[0])
        loss.backward()
        _opt.step()
        batch_time.update(time.time()-end)
        end = time.time()
        if index % 100 == 0:
            progress.display(index)
    epoch_time.update(time.time()-start)
    print("\033[1m;32{}\033[0m".format("the final output is as followed"))
    progress.display(len(_train_loader))

def save_checkpoint(state, is_best, args, filename="checkpoint.pth.tar"):
    file_path = os.path.join(args.model_path, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(args.model_path, "model_best.pth.tar"))

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_worker():
    pass

def main():
    args = parse.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    ##############################
    ####### 使用ConvMixer #########
    ##############################

    model = SmCoModel(base_encoder=ConvMixer,
                      input_dim=args.input_dim,
                      feature_dim=args.output_dim,
                      K=args.k_queue,
                      m=args.m,
                      t=args.t,
                      mlp=args.mlp).cuda(0)

    ##############################
    ####### 使用ResNet50 ##########
    ##############################

    # model = SmCoModel(base_encoder=models.__dict__[args.arch],
    #                   input_dim=args.input_dim,
    #                   feature_dim=args.output_dim,
    #                   K=args.k_queue,
    #                   m=args.m,
    #                   t=args.t,
    #                   mlp=args.mlp).cuda(0)

    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("\033[33;1m{}\033[0m".format("----loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("\033[33;1m{}\033[0m".format("----loading checkpoint and start from {}".format(args.start_epoch)))
        else:
            print("\033[33;1m{}\033[0m".format("no checkpoint has been found at {}").format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 不同版本的数据增强
    if args.aug_plus == 0:
        # MoCo V1 版本数据增强 the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.aug_plus == 1:
        # MoCo V2版本数据增强 similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.aug_plus == 2:
        pass

    train_datasets = datasets.CIFAR10("D:\\Datasetsd\\DataSets\\CIFAR\\CIFAR10", download=True, train=True,
                                      transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_loader = DataLoader(dataset=train_datasets, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True)

    for e in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer=optimizer, epoch=e, args=args)
        train(_train_loader=train_loader,
              _epoch=e,
              _model=model,
              _opt=optimizer,
              args=args)
        save_checkpoint(state={
            'epoch': e + 1,
            'arch' : args.arch,
            'state_dict':model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }, is_best=False, filename="checkpoint_{:04d}.pth.tar".format(e), args=args)

if __name__ == '__main__':
    main()