import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseEncoder import Encoder
import einops
import numpy as np
from my_utils import priorityQueue_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SmCoModel(nn.Module):
    def __init__(self, base_encoder, input_dim = 1, feature_dim = 10, K = 128, m = 0.999, t = 0.07, mlp = False):
        """

        :param base_encoder: encoder obj
        :type base_encoder:
        :param input_dim: encoder mdoel input dimension  for initial the encoder model
        :type input_dim:
        :param feature_dim:encoder model output dimension, and also the embedding feature's number
        :type feature_dim:
        :param K: the length of the queue
        :type K:
        :param m:the momentum parameter
        :type m:
        :param t:the softmax temperature
        :type t:
        :param mlp:judge if the fc layer is only one
        :type mlp:
        """
        super(SmCoModel, self).__init__()
        self.K = K
        self.m = m
        self.t = t

        self.encoder_q = base_encoder(_input_dim = input_dim, _output_dim = feature_dim)
        self.encoder_k = base_encoder(_input_dim = input_dim, _output_dim = feature_dim)

        if mlp:# 判定是否需要增加全连接层 正常需要增加有两个全连接层
            input_mlp = self.encoder_q.fc.weight.shape[1]# weight的维度为：output x input
            print("input dimension is ", self.encoder_q.fc.weight.shape)
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(in_features=input_mlp, out_features=input_mlp),
                nn.ReLU(inplace=True),
                self.encoder_q.fc

            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(in_features=input_mlp, out_features=input_mlp),
                nn.ReLU(inplace=True),
                self.encoder_k.fc
            )

        # 初始化权重 k的权重值是由q决定的 并且无反向传播
        for param_q, param_k  in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q)
            param_k.requires_grad = False

        # 定义queue  queue初始化的时候 是通过在0到1内随机正态分布采样进行初始化
        self.register_buffer("queue", torch.randn([feature_dim, self.K]))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))# 指针 操控队列

    # 三种权重生成算法
    # 计算q和k_queue的权重 相似度越小/距离越大 权重越大 相似度越大/距离越小 权重越小
    @torch.no_grad()
    def _weight_method_euclidean(self, q):
        """

        :param q: q_embedding
        :type q: [bs, c]
        :param self.queue: the queue of k_embedding
        :type self.queue:[c, k]
        :return:weight
        :rtype:[bs, queue_length]
        """
        bs = q.size()[0]
        k_length = self.queue.shape[-1]
        k_queue = self.queue.T # [128, 10]->[k, c]

        # xq [bs, queue_len, feature_num]
        xq = einops.repeat(q, "b c->b n c", n = k_length)

        # xk [bs, queue_len, feature_num]
        xk = einops.repeat(k_queue, "k c->n k c", n=bs)

        # all_embedding_eluc [bs, queue_len]  [i, j] 代表 第i个batch中 第k_queue[j]与q的欧式距离
        all_embedding_eluc = torch.norm(xq-xk, p=2, dim=-1)
        weight = 1/(1+all_embedding_eluc)
        # print(f"the weight is{weight}, and the shape is {weight.shape} ")
        return weight



    @torch.no_grad()
    def _weight_method_cos_similarity(self, q):
        """

        注意一点 相似度代表的是相似程度 常用余弦相似度 余弦相似度相似度越大 代表越相似 但是这里越相似我们需要的权重值越小
        所以不能用余弦相似度 这里采用的是正弦相似度 正弦相似度越大 代表越不相似 和所需权重相吻合
        :param q: q_embedding
        :type q: [bs, c]
        :param self.queue: the queue of k_embedding
        :type self.queue:[c, k]
        :return:weight
        :rtype:[bs, queue_length]
        """
        bs = q.size()[0]
        k_length = self.queue.shape[-1]
        k_queue = self.queue.T  # [128, 10]->[k, c]

        # xq [bs, queue_len, feature_num]
        xq = einops.repeat(q, "b c->b n c", n=k_length)

        # xk [bs, queue_len, feature_num]
        xk = einops.repeat(k_queue, "k c->n k c", n=bs)
        cos_sim = torch.cosine_similarity(xq, xk, dim=-1)
        # print(f"cos_sim shape is{cos_sim.shape}, and the num is {cos_sim}")
        # 计算正弦相似度
        sin_sim = 1-cos_sim**2
        # print(f"weight shape is{sin_sim.shape}, and the num is {sin_sim}")
        weight = sin_sim
        # 归一化处理
        # weight = .5+.5*cos_sim
        return weight

    @torch.no_grad()
    def _dijkstra(self, adj):
        n = adj.size()[0]
        distance = np.inf * torch.ones(n)
        distance[0] = 0
        q = priorityQueue_torch(0)
        while not q.is_Empty():
            v, dv = q.top()
            v = int(v)
            q.pop()
            if dv != distance[v]:
                continue
            for i, weight in enumerate(adj[v]):
                if weight < 0:
                    continue
                if weight == 0 and i != v:
                    continue
                else:
                    to = i
                    if distance[v] + weight < distance[to]:
                        distance[to] = distance[v] + weight
                        q.push(torch.tensor([to, distance[to]]))

        return distance

    @torch.no_grad()
    def _weight_method_isomap(self, q, n_node=5):
        """

        :param q: query embedding
        :type q: [bs c]
        :param n_node: the k num that we wanna get to approximate the eluc distance
        :type n_node: int
        :return: weight
        :rtype: [bs, queue_length]
        """
        bs = q.size()[0]
        k_queue_t = self.queue.T#[k, c]
        xk = einops.repeat(k_queue_t, "k c->n k c", n=bs)# [bs k c]
        xq = q.unsqueeze(1)# [bs 1 c]
        all = torch.cat([xq, xk], dim=1)# [bs, k+1, c]
        # print(all.shape)
        new_k_len = self.K+1
        xall = einops.repeat(all, "b k c->b k n c", n=new_k_len)
        yall = einops.repeat(all, "b k c->b n k c", n=new_k_len)
        all_eluc = torch.norm(xall-yall, p=2, dim=-1)
        value, index = torch.topk(input=all_eluc, k=new_k_len-n_node, dim=-1, sorted=True, largest=False)
        source = torch.zeros(all_eluc.shape).to(device)
        source -= 1
        new_all_eluc = torch.scatter(dim=2, index=index, input=all_eluc, src=source)
        distance = None
        for i in range(bs):
            if distance is None:
                distance = self._dijkstra(new_all_eluc[i])
            else:
                distance = torch.vstack([distance, self._dijkstra(new_all_eluc[i])])
        # print(distance)
        distance = distance[:, 1:]
        new_distance = torch.where(torch.isinf(distance), torch.full_like(distance, 0), distance)
        mx = torch.max(new_distance)
        new_distance = torch.where(torch.isinf(distance), torch.full_like(distance, mx+1), distance)
        weight = 1/(1+new_distance).to(device)
        # print(weight.shape)
        # [bs k]
        return weight

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _update_queue_weight(self, weight, bs):
        """
        Update the queue weight as weighted neg sample
        weight shape is [bs k]
        """
        # assert list(weight.shape)[0]==self.K

        x_queue = einops.repeat(self.queue, "c k->n c k", n=bs)# x_queue[bs c k]
        bs_weighted_queue = torch.einsum("bck,bk->bck", [x_queue, weight])
        # shape is [bs c k]
        return bs_weighted_queue

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bs = keys.size()[0]
        ptr = int(self.queue_ptr)
        assert self.K % bs == 0
        self.queue[:, ptr:ptr+bs] = keys.T
        ptr = (ptr + bs) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _generate_weight(self, q, method="e"):
        """
        generate the weight of the queue's neg vec
        :return:weight
        :rtype:[K]
        "e" is _weight_method_euclidean()
        """
        # return torch.randn([4, self.K]).to(device=device)
        # weight = self._weight_method_euclidean(q)
        weight = self._weight_method_isomap(q)
        return weight

    def forward(self, img_q, img_k):
        """

        :param img_k: a batch of key image
        :type img_k: [bs channels w h]
        :param img_q: a batch of query image
        :type img_q: [bs channels w h]
        :return: logits label
        :rtype:
        """
        bs = img_q.size()[0]
        q = self.encoder_q(img_q)
        q = F.normalize(q, dim=-1)

        # self._weight_method_isomap(q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(img_k )
            k = F.normalize(k, dim=-1)
            weight = self._generate_weight(q)
            # shape[bs c k]
            bs_weighted_queue = self._update_queue_weight(weight=weight, bs=bs)#[]
        # l_pos = [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_neg = [N, K]
        l_neg = torch.einsum('nc,nck->nk', [q, bs_weighted_queue.detach()])

        # logits = [N, K+1]
        logits = torch.cat([l_pos, l_neg], dim=1)
        # temperature
        logits /= self.t

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        #dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels



# def main():
#     image = torch.randn([4, 1, 28, 28]).to(device=device)
#     # model = Encoder(_input_dim=1, _output_dim=10).to(device=device)
#     smco_model = SmCo(Encoder, input_dim=1, feature_dim=10).to(device=device)
#     logits, labels = smco_model(image, image)
#     # print(f"logits is {logits}, labels is {labels}")
#     print(f"logits shape is {logits.shape}, label shape is {labels.shape}")
#
#
#
# if __name__ == '__main__':
#     main()