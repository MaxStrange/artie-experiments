"""
Noise functions taken from https://www.kaggle.com/code/huseinzol05/sound-augmentation-librosa/notebook
"""
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import librosa
import logging
import math
import numpy as np
import random
import torch
import torchaudio

class ToneDataset(IterableDataset):
    """
    ToneDataset is the base class for all types of tone datasets.
    """
    def __init__(self, config) -> None:
        super().__init__()

        # Read in configuration file
        self._duration_seconds = config.getfloat('Dataset', 'signal-duration-seconds')
        self._sample_rate_hz = config.getfloat('Dataset', 'sample-rate-hz')
        self._size = config.getint('Dataset', 'size')
        self._hop_length = config.getint('Dataset', 'hop-length')
        self._normalize = config.getbool('Dataset', 'normalize')
        self._toneclasses = config.getlist('Dataset', 'tones', type=float)

        # Create transformation pipeline
        self._transform = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(n_fft=config.getint('Dataset', 'num-samples-per-fft'), hop_length=self._hop_length, normalized=self._normalize),
        )

        # Set up internal state
        self._current_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._size

    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process is handling all the data loading
            return self._single_process_next()
        else:
            # I am a worker process; the dataset is partitioned across me and my brethren
            return self._worker_process_next(worker_info)

    def _single_process_next(self):
        if self._current_index >= len(self._indexes):
            self._current_index = 0
            raise StopIteration
        else:
            item = self[self._current_index]
            self._current_index += 1
            return item

    def _worker_process_next(self, worker_info):
        items_per_worker = int(math.ceil(self._size / float(worker_info.num_workers)))
        worker_id = worker_info.id
        my_start = worker_id * items_per_worker
        my_end = min(my_start + items_per_worker, self._size)
        if self._current_index < my_start:
            self._current_index = my_start

        if self._current_index >= my_end:
            self._current_index = my_start
            raise StopIteration
        else:
            item = self[self._current_index]
            self._current_index += 1
            return item


class PureToneDataset(ToneDataset):
    """
    A ToneDataset where the tones are pure. We assume we will only have a few tones
    and so we generate this dataset from scratch each time we construct it.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        # Generate the dataset
        # One index into the _items array for each item in the dataset
        tones_indexes = [i for i in range(len(self._toneclasses))]
        self._indexes = [random.choice(tones_indexes) for _ in range(self._size)]
        # One copy of each type of item - i.e., one (spectrogram, label) for each type of spectrogram
        self._items = [self._generate_pure_tone_spectrogram(tone, class_index) for (class_index, tone) in enumerate(self._toneclasses)]

    def __getitem__(self, index):
        # index -> self._indexes -> self._items
        item_index = self._indexes[index]
        return self._items[item_index]

    def _generate_pure_tone_spectrogram(self, freqhz: float, class_index: int):
        # Generate a pure tone
        npwaveform = librosa.tone(freqhz, sr=self._sample_rate_hz, duration=self._duration_seconds)
        torchwaveform = torch.from_numpy(npwaveform)
        spectrogram = self._transform(torchwaveform)
        if self._normalize:
            spectrogram /= torch.max(spectrogram)

        # Unsqueeze to add a channels dimension
        spectrogram = torch.unsqueeze(spectrogram, 0)

        return spectrogram, (class_index, freqhz)

class NoisyToneDataset(ToneDataset):
    """
    A set of tones, but the underlying tone is noisy. The model is expected
    to recover the underlying tone from the noisy example.

    We generate the original tones up front, but each time a data point
    is requested, we add noise to it on the fly.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self._noise_functions = []
        noise_types = [s for s in config.getlist('Dataset', 'noise-types', type=str) if s != ""]
        for noise_type in noise_types:
            match noise_type:
                case "pitch-and-speed":
                    self._noise_functions.append(self._noise_pitch_and_speed)
                case "pitch":
                    self._noise_functions.append(self._noise_pitch)
                case "speed":
                    self._noise_functions.append(self._noise_speed)
                case "value-augmentation":
                    self._noise_functions.append(self._noise_value_augmentation)
                case "distribution":
                    self._noise_functions.append(self._noise_distribution)
                case "random-shifting":
                    self._noise_functions.append(self._noise_random_shifting)
                case "hpss":
                    self._noise_functions.append(self._noise_hpss)
                case "stretch":
                    self._noise_functions.append(self._noise_stretch)
                case _:
                    errmsg = f"Cannot interpret {noise_type} as a type of noise to add in the data pipeline"
                    logging.error(errmsg)
                    raise ValueError(errmsg)

        # Generate the dataset
        # One index into the _items array for each item in the dataset
        tones_indexes = [i for i in range(len(self._toneclasses))]
        self._indexes = [random.choice(tones_indexes) for _ in range(self._size)]
        # One copy of each type of item - i.e., one (spectrogram, label) for each type of spectrogram
        self._items = [self._generate_pure_tone(tone, class_index) for (class_index, tone) in enumerate(self._toneclasses)]

    def __getitem__(self, index):
        item_index = self._indexes[index]
        x, y = self._items[item_index]
        x = self._apply_noise_pipeline(x)
        x = torch.from_numpy(x)
        spectrogram = self._transform(x)
        if self._normalize:
            spectrogram /= torch.max(spectrogram)
        spectrogram = torch.unsqueeze(spectrogram, 0)  # Add color channel
        return spectrogram, y

    def _generate_pure_tone(self, freqhz: float, class_index: int):
        # Generate a pure tone
        npwaveform = librosa.tone(freqhz, sr=self._sample_rate_hz, duration=self._duration_seconds)
        return npwaveform, (class_index, freqhz)

    def _apply_noise_pipeline(self, waveform: np.ndarray) -> np.ndarray:
        for noise_function in self._noise_functions:
            waveform = noise_function(waveform)
        return waveform

    def _noise_pitch_and_speed(self, waveform: np.ndarray) -> np.ndarray:
        y_pitch_speed = waveform.copy()
        length_change = np.random.uniform(low=0.8, high = 1)
        speed_fac = 1.0  / length_change
        tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)
        minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
        y_pitch_speed *= 0
        y_pitch_speed[0:minlen] = tmp[0:minlen]
        return y_pitch_speed

    def _noise_pitch(self, waveform: np.ndarray) -> np.ndarray:
        bins_per_octave = 12
        pitch_pm = 2
        pitch_change =  pitch_pm * 2*(np.random.uniform())   
        y_pitch = librosa.effects.pitch_shift(waveform.astype('float64'), sr=self._sample_rate_hz, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        return y_pitch

    def _noise_speed(self, waveform: np.ndarray) -> np.ndarray:
        speed_change = np.random.uniform(low=0.9,high=1.1)
        tmp = librosa.effects.time_stretch(waveform.astype('float64'), rate=speed_change)
        minlen = min(waveform.shape[0], tmp.shape[0])
        waveform *= 0 
        waveform[0:minlen] = tmp[0:minlen]
        return waveform

    def _noise_value_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        dyn_change = np.random.uniform(low=1.5,high=3)
        y_aug = waveform * dyn_change
        return y_aug

    def _noise_distribution(self, waveform: np.ndarray) -> np.ndarray:
        #noise_amp = 0.005*np.random.normal()*np.amax(waveform)
        noise_amp = 0.005*np.random.uniform()*np.amax(waveform)
        y_noise = waveform.astype('float64') + noise_amp * np.random.normal(size=waveform.shape[0])
        return y_noise

    def _noise_random_shifting(self, waveform: np.ndarray) -> np.ndarray:
        timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
        start = int(waveform.shape[0] * timeshift_fac)
        if (start > 0):
            waveform = np.pad(waveform,(start,0),mode='constant')[0:waveform.shape[0]]
        else:
            waveform = np.pad(waveform,(0,-start),mode='constant')[0:waveform.shape[0]]
        return waveform

    def _noise_hpss(self, waveform: np.ndarray) -> np.ndarray:
        # Harmonic and percussive. Let's return just the harmonic
        harmonic, percussive = librosa.effects.hpss(waveform.astype('float64'))
        return harmonic

    def _noise_stretch(self, waveform: np.ndarray) -> np.ndarray:
        input_length = len(waveform)
        stretching = librosa.effects.time_stretch(waveform.astype('float'), rate=1.1)
        if len(stretching) > input_length:
            stretching = stretching[:input_length]
        else:
            stretching = np.pad(stretching, (0, max(0, input_length - len(stretching))), "constant")
        return stretching


def make_from_config_file(config):
    assert config.getstr('Dataset', 'type').lower() == "classifier"
    noisy = len([s for s in config.getlist('Dataset', 'noise-types', type=str) if s != ""]) > 0
    if noisy:
        traindataset = NoisyToneDataset(config)
        testdataset = NoisyToneDataset(config)
        valdataset = NoisyToneDataset(config)
    else:
        traindataset = PureToneDataset(config)
        testdataset = PureToneDataset(config)
        valdataset = PureToneDataset(config)

    batchsize = config.getint('Dataset', 'batch-size')
    nworkers = config.getint('Dataset', 'num-workers')
    if nworkers > 0:
        train = DataLoader(traindataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
        test = DataLoader(testdataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
        val = DataLoader(valdataset, batch_size=batchsize, num_workers=nworkers, prefetch_factor=4)
    else:
        train = DataLoader(traindataset, batch_size=batchsize)
        test = DataLoader(testdataset, batch_size=batchsize)
        val = DataLoader(valdataset, batch_size=batchsize)

    return train, val, test