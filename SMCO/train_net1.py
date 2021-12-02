import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import argparse
import os
import random
import loader
from BaseEncoder import Encoder
from SmCo import SmCoModel
import torch.nn.functional as F

parse = argparse.ArgumentParser(description="this is the test")
parse.add_argument("--batch-size", default=16, type=int,
                   help="batch size")
parse.add_argument("--epoch", default=100, type=int,
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
parse.add_argument("--mlp", default=True,
                   help="if need the mlp to do the lincl")
parse.add_argument("--output-dim", default=10, type=int,
                   help="the encoder output dimention")
parse.add_argument("--lr", default=0.0025, type=float,
                   help="learning rate")
parse.add_argument("--momentum", default=0.9, type=float,
                   help="the momentum of the SGD")
parse.add_argument("--weight-decay", default=1e-4, type=float,
                   help="the SGD weight decay")

class AverageMeter(object):
    def __init__(self, name, fmt = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum//self.cnt

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size()[0]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k*100/batch_size*k)
        return res

def train(_model, _opt, _epoch, _train_loader, args):
    # 参数初始化部分
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(_train_loader),
        [losses, top1, top5],
        prefix="Epoch:[{}]".format(_epoch)
    )
    # 模型训练部分
    _model.train()
    for index, (images, label) in enumerate(_train_loader):
        images[0], images[1] = images[0].to(device=args.device), images[1].to(device=args.device)
        label = label.to(device=args.device)
        _opt.zero_grad()
        output, target = _model(img_q = images[0], img_k=images[1])
        loss = F.cross_entropy(output, target)
        # _, pred = output.topk(5, 1, True, True)
        # print(pred.shape)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)
        # print(correct.shape)
        # res = []
        # correct_k = correct[:5].contiguous().view(-1).float().sum(0, keepdim=False)
        # print("correct_k shape is ", correct_k.shape)
        # print(correct_k)
        acc1, acc5 = accuracy(output=output, target=target, topk=(1, 5))
        losses.update(loss.item(), images[0].size()[0])
        top1.update(acc1[0], images[0].size()[0])
        top5.update(acc5[0], images[0].size()[0])
        loss.backward()
        _opt.step()



def main():
    args = parse.parse_args()
    # 设置随机数种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_worker(args)

def main_worker(args):
    print("\033[32;1m{}\033[0m".format("start the main work"))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    arguments = augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    train_datasets = datasets.CIFAR10("D:\\Datasetsd\\DataSets\\CIFAR\\CIFAR10", download=True, train=True,
                                      transform=loader.TwoCropsTransform(transforms.Compose(arguments)))
    test_datasets = datasets.CIFAR10("D:\\Datasetsd\\DataSets\\CIFAR\\CIFAR10", download=True, train=False,
                                      transform=loader.TwoCropsTransform(transforms.Compose(arguments)))
    # print(train_datasets)
    train_loader = DataLoader(dataset=train_datasets, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_datasets, shuffle=True, batch_size=args.batch_size)

    model = SmCoModel(base_encoder=Encoder,
                      input_dim=args.input_dim,
                      feature_dim=args.output_dim,
                      K=args.k_queue,
                      m=args.m,
                      t=args.t,
                      mlp=args.mlp).cuda(0)
    print("\033[32;1m{}\033[0m".format("the model structure is as followed"))
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    for e in range(args.epoch):
        train(_model=model, _opt=optimizer, _epoch=e, _train_loader=train_loader, args=args)





if __name__ == '__main__':
    main()