"""
Dataset code specific to clustering experiments.
"""
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import pysinewave
import torch
import torchaudio


class ClusteringDataset(IterableDataset):
    def __init__(self, config) -> None:
        super(ClusteringDataset).__init__()
        self._size = int(10 * config.getint('Dataset', 'batch-size'))
        toneclasses = config.getlist('Dataset', 'tones', type=float)
        self._items = []
        for i in range(self._size):
            spectrogram = self._generate_pure_tone_spectrogram(toneclasses[i % len(toneclasses)])
            self._items.append(spectrogram)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return self._size

    def _generate_pure_tone_spectrogram(freqhz: float):
        wave = pysinewave.SineWave()
        wave.set_frequency(freqhz)
        wave.set_volume(50)
        # Read the wave into Torchaudio
        # Turn the audio into a spectrogram
        # Tensorify the spectrogram

def make_dataset_from_config_file(config):
    """
    Make a clustering experiment's dataset based on the given configuration file.
    """
    dataset = ClusteringDataset(config)
    batchsize = config.getint('Dataset', 'batch-size')
    train = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    return train, None