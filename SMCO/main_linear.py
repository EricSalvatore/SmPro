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
parse.add_argument("--resume", default="", type=str,
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

