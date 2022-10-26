"""
Dataset code specific to clustering experiments.
"""
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm
import logging
import pysinewave
import torch
import torchaudio

class ClusteringDataset(IterableDataset):
    def __init__(self, config) -> None:
        super(ClusteringDataset).__init__()
        self._duration_seconds = config.getfloat('Dataset', 'signal-duration-seconds')
        self._sample_rate_hz = config.getfloat('Dataset', 'sample-rate-hz')
        self._size = config.getint('Dataset', 'size')
        toneclasses = config.getlist('Dataset', 'tones', type=float)

        logging.info("Generating dataset...")
        self._items = []
        for i in tqdm(range(self._size)):
            spectrogram, label = self._generate_pure_tone_spectrogram(toneclasses[i % len(toneclasses)])
            self._items.append((spectrogram, label))

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return self._size

    def _generate_pure_tone_spectrogram(self, freqhz: float):
        wave = pysinewave.SineWave(samplerate=self._sample_rate_hz)
        wave.set_frequency(freqhz)
        wave.set_volume(50)
        nframes = int(self._duration_seconds * self._sample_rate_hz)
        npwaveform = wave.sinewave_generator.next_data(nframes)
        # Turn the audio into a spectrogram
        # Tensorify the spectrogram
        return npwaveform, freqhz


def make_dataset_from_config_file(config):
    """
    Make a clustering experiment's dataset based on the given configuration file.
    """
    dataset = ClusteringDataset(config)
    batchsize = config.getint('Dataset', 'batch-size')
    train = DataLoader(dataset, batch_size=batchsize)
    return train, None