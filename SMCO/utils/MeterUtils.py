class AverageMeter(object):
    # 以相应格式存储loss和accuracy
    def __init__(self, name, fmt=":f"):
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
        self.avg = self.sum // self.cnt

    def __str__(self):
        fmtstr = "{name} {val" +self.fmt+ "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    # 打印输出log
    def __init__(self, num_batches, meters, prefix=""):
        """

        :param num_batches: batchsize
        :type num_batches:
        :param meters: output parameters
        :type meters:
        :param prefix:
        :type prefix:
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) #将所有batch输出的时候右对齐
        fmt = "{:" + str(num_digits) +"d}"
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
