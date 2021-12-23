import torch
import torch.nn as nn
import numpy as np

class priorityQueue_torch(object):
    def __init__(self, node):
        """
        优先队列的初始化
        :param node: the value of the node
        :type node:
        """
        # 优先队列的结点 采用pair表示
        # pair<to, dis> 表示的是从source结点到to结点的路径长度为dis
        # top = self.prior_queue[0]
        # is_Empty = if self.prior_queue.shape[0] == 0
        self.prior_queue = torch.tensor([[node, 0]])

    def push(self, x):
        """
        入堆
        :param x: 入栈结点 pair<to, dis>
        :type x:
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if self.is_Empty():
            self.prior_queue = x.unsqueeze(dim=0)
            return
        idx = torch.searchsorted(self.prior_queue.T[1].contiguous(), x[1])
        # print("\033[32;1m{}\033[0m".format("结点所在的index为{}".format(idx)))
        self.prior_queue = torch.vstack([self.prior_queue[0:idx], x, self.prior_queue[idx:]]).contiguous()

    def pop(self):
        """
        出堆
        :return:
        :rtype:
        """
        if self.is_Empty():
            print("\033[31;1m{}\033[0m".format("无法出队列，当前栈为空"))
        self.prior_queue = self.prior_queue[1:]

    def top(self):
        return self.prior_queue[0]

    def is_Empty(self):
        return self.prior_queue.shape[0] == 0

