import torch
import torch.nn as nn
import numpy as np
from my_utils import priorityQueue_torch
import einops

def dijkstra(adj):
    n = adj.size()[0]
    distance = np.inf*torch.ones(n)
    distance[0] = 0
    q = priorityQueue_torch(0)
    while not q.is_Empty():
        v, dv = q.top()
        v = int(v)
        q.pop()
        if dv != distance[v]:
            continue
        for i, weight in enumerate(adj[v]):
            if weight<0:
                continue
            if weight == 0 and  i != v:
                continue
            else:
                to = i
                if distance[v] + weight < distance[to]:
                    distance[to] = distance[v] + weight
                    q.push(torch.tensor([to, distance[to]]))

    return distance

adj = torch.tensor([[1, 2, 3, 0, -1],
                    [4, 1, 3, 5, 1],
                    [0, -1, 2, 3, 4],
                    [3, 1, 4, 1, 0],
                    [4, 2, 1, 4, -1]])

new_adj = einops.repeat(adj, "a b->n a b", n=3)

q = torch.randn([4, 3])# [bs, c]
k_queue = torch.randn([3, 10])# [c, k]
k_queue_t = k_queue.T# [k, c]
xk = einops.repeat(k_queue_t, "k c->n k c", n=4)#[bs k c]
xq = q.unsqueeze(1)# [bs 1 c]
# print(xq.shape)
all = torch.cat([xq, xk], dim=1)# [bs k+1 c]
# print(all.shape)
xall = einops.repeat(all, "b k c->b k n c", n=11)
ball = einops.repeat(all, "b k c->b n k c", n=11)
all_eluc = torch.norm(xall-ball, p=2, dim=-1)
# zero_all_eluc = all_eluc[:, 0,:]
# print(zero_all_eluc)
# print(all_eluc.shape)# [4, 11, 11] q 在0维度
value, index = torch.topk(input=all_eluc, k=11-5, dim=-1, sorted=True, largest=False)
# print("value is {}, and the shape is {}".format(value, value.shape))
# print("index is {}, and the shape is {}".format(index, index.shape))
source = torch.zeros(all_eluc.shape)
source -= 1
new_all_eluc = torch.scatter(dim=2, index=index, input=all_eluc, src=source)

distance = None
bs = all_eluc.size()[0]
for index in range(bs):
    if distance is None:
        distance = dijkstra(new_all_eluc[index])
    else:
        distance = torch.vstack([distance, dijkstra(new_all_eluc[index])])
print(distance)
print(distance.shape)
distance = distance[:, 1:]
print(distance)
print(distance.shape)
# mx = torch.max(distance)
# print(mx)
new_distance = torch.where(torch.isinf(distance), torch.full_like(distance, 0), distance)
print(new_distance)
mx = torch.max(new_distance)
print(mx)
new_distance = torch.where(torch.isinf(distance), torch.full_like(distance, float(mx+1)), distance)
print(new_distance)
weight = 1/(1+new_distance)
print("weight is ", weight)
