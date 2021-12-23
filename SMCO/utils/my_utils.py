import torch

# 正确率计算
def accuracy(output, target, topk=(1, )):
    # 计算topk的准确率  默认为1
    # target size [bs, 1]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size()[0]
        value, pred = torch.topk(input=output, dim=1, largest=True, sorted=True, k=maxk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100/(batch_size)))
        # print(res)
        return res
