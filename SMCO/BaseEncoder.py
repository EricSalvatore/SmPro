import torch
import torch.nn as nn
from Residual import ResidualNet

class ConvMixerLayer(nn.Module):
    """
        hyper-parameters:
            image patches channels--- token_ch
            image patches sizes--- p
            the kernel size of the depthwise convolution--- dp_size
        """
    def __init__(self, _token_ch, _dp_size=9):
        super(ConvMixerLayer, self).__init__()
        self.dim = _token_ch
        self.dp_size = _dp_size
        # Depthwise Convolution
        self.res_layer = ResidualNet(nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=self.dp_size,
                      groups=self.dim, padding=int((self.dp_size - 1) / 2)),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        ))
        # Pointwise Convlution
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )

    def forward(self, x):
        x = self.res_layer(x)
        # print(x.shape)
        output = self.layer2(x)
        return output

class ConvMixer(nn.Module):
    """
    Build a Encoder to Transform an image to an embedding
    input : bs x channels x w x h
    output: bs x c
    """
    def __init__(self, _input_dim=3, num_classes=10, _token_ch=100, _depth=12, _patch_size=7, _dp_size=9):
        """
        when identity an model object, you need two args to in put

        :param _input_dim: the input_channels
        :type _input_dim:
        :param _output_dim: the embedding features
        :type _output_dim:
        """
        super(ConvMixer, self).__init__()
        self.input_dim = _input_dim
        self.output_dim = num_classes
        # 从这里开始替换
        self.dim = _token_ch
        self.patch_size = _patch_size
        self.dp_size = _dp_size

        # patch_embedding
        self.embedding = nn.Conv2d(in_channels=self.input_dim, out_channels=self.dim,
                                   kernel_size=self.patch_size, stride=self.patch_size)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim)
        self.ConvMixerList = nn.ModuleList([])
        for _ in range(_depth):
            self.ConvMixerList.append(ConvMixerLayer(_token_ch=self.dim, _dp_size=self.dp_size))
        self.adp = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.dim, out_features=self.output_dim)

    def forward(self, x):
        bs = x.size()[0]
        x = self.embedding(x)
        x = self.gelu(x)
        x = self.bn(x)
        for conmix_layer in self.ConvMixerList:
            x = conmix_layer(x)
        x = self.adp(x)
        x = x.view(bs, -1)
        output = self.fc(x)
        return output

#
# class Encoder(nn.Module):
#     """
#     Build a Encoder to Transform an image to an embedding
#     input : bs x channels x w x h
#     output: bs x c
#     """
#     def __init__(self, _input_dim, _output_dim):
#         """
#         when identity an model object, you need two args to input
#
#         :param _input_dim: the input_channels
#         :type _input_dim:
#         :param _output_dim: the embedding features
#         :type _output_dim:
#         """
#         super(Encoder, self).__init__()
#         self.input_dim = _input_dim
#         self.output_dim = _output_dim
#         # 从这里开始替换
#         self.conv1 = nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=3,
#                                stride=1, padding=1)
#         self.activate_pooling = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
#                                stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(in_features=64*26*26, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=self.output_dim)
#
#     def forward(self, x):
#         bs = x.size()[0]
#         x = self.conv1(x)
#         x = self.activate_pooling(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.activate_pooling(x)
#         x = self.relu(x)
#         x = x.view(bs, -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         output = self.fc2(x)
#         return output



# if __name__ == '__main__':
#     image = torch.randn([4, 3, 28, 28]).to(device=torch.device("cuda:0"))
#     model = Encoder(_input_dim=3, _output_dim=10).to(device=torch.device("cuda:0"))
#     # model = BaseEncoder(3, 10).to(device=torch.device("cuda:0"))
#     output = model(image)
#     print(output)
