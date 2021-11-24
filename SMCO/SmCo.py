import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseEncoder import Encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SmCo(nn.Module):
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
        super(SmCo, self).__init__()
        self.K = K
        self.m = m
        self.t = t

        self.encoder_q = base_encoder(_input_dim = input_dim, _output_dim = feature_dim)
        self.encoder_k = base_encoder(_input_dim = input_dim, _output_dim = feature_dim)

        if mlp:# 判定是否需要增加全连接层 正常需要增加有两个全连接层
            input_mlp = self.encoder_q.fc.weight.shape[1]# weight的维度为：output x input
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(in_features=input_mlp, out_features=input_mlp),
                nn.ReLU(inplace=True),
                self.encoder_q.fc

            )
            self.encoder_k = nn.Sequential(
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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _update_queue_weight(self, weight):
        """
        Update the queue weight as weighted neg sample
        """
        assert list(weight.shape)[0]==self.K
        self.queue = torch.einsum("ck,k->ck", [self.queue, weight])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bs = keys.size()[0]
        ptr = int(self.queue_ptr)
        assert self.K % bs == 0
        self.queue[:, ptr:ptr+bs] = keys.T
        ptr = (ptr + bs) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _generate_weight(self):
        """
        generate the weight of the queue's neg vec
        :return:weight
        :rtype:[K]
        """
        return torch.randn([self.K]).to(device=device)

    def forward(self, img_k, img_q):
        """

        :param img_k: a batch of key image
        :type img_k: [bs channels w h]
        :param img_q: a batch of query image
        :type img_q: [bs channels w h]
        :return: logits label
        :rtype:
        """
        q = self.encoder_q(img_q)
        q = F.normalize(q, dim=-1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(img_k )
            k = F.normalize(k, dim=-1)
            weight = self._generate_weight()
            self._update_queue_weight(weight=weight)
        # l_pos = [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_neg = [N, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits = [N, K+1]
        logits = torch.cat([l_pos, l_neg], dim=1)
        # temperature
        logits /= self.t

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        #dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels



def main():
    image = torch.randn([4, 1, 28, 28]).to(device=device)
    # model = Encoder(_input_dim=1, _output_dim=10).to(device=device)
    smco_model = SmCo(Encoder, input_dim=1, feature_dim=10).to(device=device)
    logits, labels = smco_model(image, image)
    print(f"logits is {logits}, labels is {labels}")



if __name__ == '__main__':
    main()