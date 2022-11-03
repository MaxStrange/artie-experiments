"""
This module consists of the neural network architectures for the classification models.
"""
from torch import nn
from . import common
import logging


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
    nchannels = 1
    resnet = common.ResNet(nchannels)
    return nn.Sequential(resnet, nn.Linear(512, nclasses), nn.LogSoftmax(1))

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

