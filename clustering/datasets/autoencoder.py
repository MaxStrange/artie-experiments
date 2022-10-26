"""
Code related to datasets for autoencoding.
"""
from torch.utils.data import DataLoader
from . import classifier
import torch

class NoisyAutoencodingToneDataset(classifier.NoisyToneDataset):
    """
    Same as NoisyToneDataset, but we return the spectrogram along with
    a copy of itself.
    """
    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, index):
        spectrogram, (class_index, freqhz) = super().__getitem__(index)
        return spectrogram, (spectrogram, freqhz)

class PureAutoencodingToneDataset(classifier.PureToneDataset):
    """
    Same as PureToneDataset, but we return the spectrogram along
    with a copy of itself.
    """
    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, index):
        spectrogram, (class_index, freqhz) = super().__getitem__(index)
        return spectrogram, (spectrogram, freqhz)

def make_from_config_file(config):
    assert config.getstr('Dataset', 'type').lower() == "autoencoder"
    noisy = len([s for s in config.getlist('Dataset', 'noise-types', type=str) if s != ""]) > 0
    if noisy:
        traindataset = NoisyAutoencodingToneDataset(config)
        testdataset = NoisyAutoencodingToneDataset(config)
        valdataset = NoisyAutoencodingToneDataset(config)
    else:
        traindataset = PureAutoencodingToneDataset(config)
        testdataset = PureAutoencodingToneDataset(config)
        valdataset = PureAutoencodingToneDataset(config)

    batchsize = config.getint('Dataset', 'batch-size')
    nworkers = config.getint('Dataset', 'num-workers')
    if nworkers > 0:
        train = DataLoader(traindataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
        test = DataLoader(testdataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
        val = DataLoader(valdataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
    else:
        # Can't specify prefetch_factor if nworkers == 0
        train = DataLoader(traindataset, batch_size=batchsize)
        test = DataLoader(testdataset, batch_size=batchsize)
        val = DataLoader(valdataset, batch_size=batchsize)
    return train, val, test