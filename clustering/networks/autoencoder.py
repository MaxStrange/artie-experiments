from torch import nn
import logging

def _make_original_network(config):
    embedding_dims = config.getint('Network', 'embedding-dims')
    encoder = nn.Sequential(
        nn.Conv2d(1, 2, (3, 3), stride=(2, 1)),         # 1
        nn.LeakyReLU(),
        nn.Conv2d(2, 4, (3, 3), stride=(2, 2)),         # 2
        nn.LeakyReLU(),
        nn.Conv2d(4, 8, (3, 3), stride=(2, 2)),         # 3
        nn.BatchNorm2d(8),
        nn.LeakyReLU(),
        nn.Conv2d(8, 16, (3, 3), stride=(2, 2)),        # 4
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, (3, 3)),                      # 5
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, (3, 3)),                      # 6
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, (3, 3)),                     # 7
        nn.LeakyReLU(),
        nn.Conv2d(128, 256, (2, 3)),                    # 8
        nn.LeakyReLU(),
        nn.Conv2d(256, 512, (2, 3)),                    # 9
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.Conv2d(512, 512, (2, 2)),                    # 10
        nn.LeakyReLU(),
        nn.Conv2d(512, 512, (2, 2)),                    # 11
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.Flatten(),
        nn.Linear(512, embedding_dims),
        nn.LeakyReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(embedding_dims, 512),
        nn.LeakyReLU(),
        nn.Unflatten(1, (512, 1, 1)),
        nn.ConvTranspose2d(512, 512, (2, 2)),                                               # 11
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 512, (2, 2)),                                               # 10
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 512, (2, 3)),                                               # 9
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 512, (2, 3)),                                               # 8
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 512, (3, 3)),                                               # 7
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 256, (3, 3)),                                               # 6
        nn.LeakyReLU(),
        nn.ConvTranspose2d(256, 256, (3, 3)),                                               # 5
        nn.LeakyReLU(),
        nn.ConvTranspose2d(256, 128, (3, 3), stride=(2, 2), output_padding=(1, 0)),         # 4
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 2), output_padding=(1, 0)),          # 3
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), output_padding=(1, 0)),           # 2
        nn.LeakyReLU(),
        nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 1)),                                  # 1
        nn.LeakyReLU(),
        nn.Conv2d(16, 1, (5, 1)),
        nn.Sigmoid()
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

