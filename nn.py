import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numbers import Number


# Utility function to ensure input is converted to tuple
def to_tuple(v, n):
    return (v,) * n if isinstance(v, Number) else tuple(v)

# Kaiming normal initialization adapted for Objax
def kaiming_normal_init(tensor, kernel_size, in_channels, out_channels, gain=1):
    shape = (*to_tuple(kernel_size, 2), in_channels, out_channels)
    fan_in = np.prod(shape[:-1])
    std = gain * np.sqrt(1 / fan_in)
    with torch.no_grad():
        tensor.normal_(0, std)

# Initialize convolutional and linear layers using Objax's default
def initialize_layers(convs, fcs):
    for conv in convs:
        kaiming_normal_init(conv.weight, conv.kernel_size, conv.in_channels, conv.out_channels)
        nn.init.zeros_(conv.bias)
    for fc in fcs:
        nn.init.xavier_normal_(fc.weight)
        nn.init.zeros_(fc.bias)

# VGG architecture for CIFAR-10
class VGG(nn.Module):
    def __init__(self, nclass, dense_size, activation=torch.tanh, colors=3):
        super(VGG, self).__init__()
        self.activation = activation
        self.conv_layers = nn.ModuleList([nn.Conv2d(colors, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1),
                                          nn.Conv2d(32, 64, 3, padding=1), nn.Conv2d(64, 64, 3, padding=1),
                                          nn.Conv2d(64, 128, 3, padding=1), nn.Conv2d(128, 128, 3, padding=1)])
        self.fc1 = nn.Linear(128 * 16, dense_size)
        self.fc2 = nn.Linear(dense_size, nclass)
        initialize_layers(self.conv_layers, [self.fc1, self.fc2])

    def forward(self, x):
        for conv in self.conv_layers[:2]: x = self.activation(conv(x))
        x = F.max_pool2d(x, 2)
        for conv in self.conv_layers[2:4]: x = self.activation(conv(x))
        x = F.max_pool2d(x, 2)
        for conv in self.conv_layers[4:]: x = self.activation(conv(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc2(self.activation(self.fc1(x)))

# Simple architecture for MNIST/EMNIST
class SmallNN(nn.Module):
    def __init__(self, nclass=10):
        super(SmallNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, nclass)
        initialize_layers([self.conv1, self.conv2], [self.fc1, self.fc2])

    def forward(self, x):
        x = F.max_pool2d(torch.tanh(self.conv1(x)), 2, 1)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2, 1)
        x = torch.tanh(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

# Function to return the appropriate model
def get_nn(model_name, nclass, colors=3):
    return VGG(nclass, dense_size=int(model_name[3:]), colors=colors) if model_name.startswith('vgg') else SmallNN(nclass)
