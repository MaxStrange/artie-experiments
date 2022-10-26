"""
This module consists of the neural network architectures for the classification models.
"""
from torch import nn
from torch import functional as F
import logging

class ResidualBlock(nn.Module):
    """
    Classic Residual block. Stolen shamelessly from
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    under MIT license.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    Simple ResNet stolen shamelessly from
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    under MIT license (then adjusted for our use case).
    """
    def __init__(self, ninput_channels, num_classes):
        super().__init__()
        self.in_channels = 16  # Starting number of feature maps
        self.conv = nn.Conv2d(ninput_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_res_block(self.in_channels, 2)
        self.layer2 = self.make_res_block(32, 2, 2)
        self.layer3 = self.make_res_block(64, 2, 2)
        self.avg_pool = nn.AvgPool2d(12)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        self.outputlayer = nn.LogSoftmax(1)
        
    def make_res_block(self, out_channels, nblocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # First block may need to downsample
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        for _ in range(1, nblocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        self.in_channels = out_channels  # Update feature map dimension
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.outputlayer(out)
        return out


def _make_original_network(config):
    toneclasses = config.getlist('Dataset', 'tones', type=float)
    nclasses = len(toneclasses)
    network = nn.Sequential(
        nn.Conv2d(1, 2, (75, 50)),
        nn.ReLU(),
        nn.Conv2d(2, 4, (50, 25)),
        nn.ReLU(),
        nn.Conv2d(4, 8, (50, 25)),
        nn.ReLU(),
        nn.Conv2d(8, 16, (25, 12)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, nclasses),
        nn.LogSoftmax(1)
    )
    return network

def _make_pooling_network(config):
    toneclasses = config.getlist('Dataset', 'tones', type=float)
    nclasses = len(toneclasses)
    network = nn.Sequential(
        nn.Conv2d(1, 16, (50, 25)),
        nn.ReLU(),
        nn.Conv2d(16, 32, (25, 12)),
        nn.ReLU(),
        nn.AvgPool2d((2, 2)),
        nn.Conv2d(32, 64, (12, 12)),
        nn.ReLU(),
        nn.Conv2d(64, 128, (8, 8)),
        nn.ReLU(),
        nn.AvgPool2d((2, 2)),
        nn.Conv2d(128, 256, (4, 4)),
        nn.ReLU(),
        nn.AvgPool2d((4, 4)),
        nn.AvgPool2d((5, 1)),
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, nclasses),
        nn.LogSoftmax(1)
    )
    return network

def _make_residual_network(config):
    toneclasses = config.getlist('Dataset', 'tones', type=float)
    nclasses = len(toneclasses)
    return ResNet(1, nclasses)

def make_from_config_file(config):
    """
    Make and return a classification neural network.

    This type of neural network is trying to classify the input spectrograms
    according to the labels.
    """
    match nntype := config.getstr('Network', 'subtype').lower():
        case "original":
            return _make_original_network(config)
        case "pooling":
            return _make_pooling_network(config)
        case "resnet":
            return _make_residual_network(config)
        case _:
            errmsg = f"Cannot interpret {nntype} as a type of neural network to make."
            logging.error(errmsg)
            raise ValueError(errmsg)

