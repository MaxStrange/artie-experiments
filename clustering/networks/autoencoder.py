from torch import nn
import logging

def _make_original_network(config):
    embedding_dims = config.getint('Network', 'embedding-dims')
    encoder = nn.Sequential(
        nn.Conv2d(1, 2, (5, 5), stride=(2, 1)),
        nn.ReLU(),
        nn.Conv2d(2, 4, (5, 5), stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(4, 8, (5, 5), stride=(2, 2)),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, (5, 5), stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4, 4)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, (2, 3)),
        nn.ReLU(),
        nn.Conv2d(128, 256, (2, 3)),
        nn.ReLU(),
        nn.Conv2d(256, 512, (2, 2)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, embedding_dims),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(embedding_dims, 512),
        nn.ReLU(),
        nn.Unflatten(1, (512, 1, 1)),
        nn.ConvTranspose2d(512, 512, (2, 2)),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 512, (2, 3)),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 512, (2, 3)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 256, (4, 4)),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, (5, 5), stride=(2, 2), output_padding=(1, 0)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, (5, 5), stride=(2, 2), output_padding=(1, 0)),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 16, (5, 5), stride=(2, 2)),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, (5, 5), stride=(2, 1)),
        nn.Sigmoid()
        # TODO: Use smaller kernels and/or use stride later in the encoder/earlier in the decoder, that way
        #       we hopefully will be able to get the bottom and the top? <-- Think about this
    )  # Need to get to 201, 113
    network = nn.Sequential(
        encoder,
        decoder
    )
    return network

def make_from_config_file(config):
    """
    Make and return a classification neural network.

    This type of neural network is trying to classify the input spectrograms
    according to the labels.
    """
    match nntype := config.getstr('Network', 'subtype').lower():
        case "original":
            return _make_original_network(config)
        case _:
            errmsg = f"Cannot interpret {nntype} as a type of neural network to make."
            logging.error(errmsg)
            raise ValueError(errmsg)

