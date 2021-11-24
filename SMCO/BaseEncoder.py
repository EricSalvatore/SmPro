import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Build a Encoder to Transform an image to an embedding
    input : bs x channels x w x h
    output: bs x c
    """
    def __init__(self, _input_dim, _output_dim):
        """
        when identity an model object, you need two args to input

        :param _input_dim: the input_channels
        :type _input_dim:
        :param _output_dim: the embedding features
        :type _output_dim:
        """
        super(Encoder, self).__init__()
        self.input_dim = _input_dim
        self.output_dim = _output_dim
        # 从这里开始替换
        self.conv1 = nn.Conv2d(in_channels=self.input_dim, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.activate_pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=64*26*26, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.output_dim)

    def forward(self, x):
        bs = x.size()[0]
        x = self.conv1(x)
        x = self.activate_pooling(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.activate_pooling(x)
        x = self.relu(x)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output



# if __name__ == '__main__':
#     image = torch.randn([4, 3, 28, 28]).to(device=torch.device("cuda:0"))
#     model = BaseEncoder(_input_dim=3, _output_dim=10).to(device=torch.device("cuda:0"))
#     # model = BaseEncoder(3, 10).to(device=torch.device("cuda:0"))
#     output = model(image)
#     print(output)
