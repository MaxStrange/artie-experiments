from torch.utils.tensorboard import SummaryWriter
import configuration
import librosa
import logging
import matplotlib.pyplot as plt
import torch

class Scheduler:
    """
    Wrapper around whatever scheduler algorithm, but it also writes to Tensorboard.
    """
    def __init__(self, scheduler, writer: SummaryWriter) -> None:
        self._scheduler = scheduler
        self._writer = writer

    def step(self, global_step: int):
        self._scheduler.step()
        lr = self._scheduler.get_last_lr()
        assert len(lr) == 1, f"Learning rate is not a list of one item: {lr}"
        lr = lr[0]
        self._writer.add_scalar("Train/learning-rate", lr, global_step)

class NoopLRScheduler:
    """
    Does nothing. Just useful so that we don't have to check for None when trying to use it.
    """
    def __init__(self, optimizer) -> None:
        self._optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return self._optimizer.lr

def make_scheduler_from_config_file(config: configuration.Configuration, optimizer, writer: SummaryWriter):
    """
    """
    scheduler_params = config.getdict('Training', 'scheduler-params', keytype=str, valuetype=float)
    match scheduler_name := config.getstr('Training', 'scheduler'):
        case "None":
            return Scheduler(NoopLRScheduler(optimizer), writer)
        case "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
            return Scheduler(scheduler, writer)
        case _:
            errmsg = f"Cannot interpret {scheduler_name} for constructing a learning rate scheduler."
            logging.error(errmsg)
            raise ValueError(errmsg)

def make_optimizer_from_config_file(config: configuration.Configuration, network: torch.nn.Module):
    """
    """
    match optimizer_name := config.getstr('Training', 'optimizer'):
        case "Adam":
            lr = config.getfloat('Training', 'learning-rate')
            return torch.optim.Adam(network.parameters(), lr=lr)
        case "SGD":
            lr = config.getfloat('Training', 'learning-rate')
            return torch.optim.SGD(network.parameters(), lr=lr)
        case _:
            errmsg = f"Cannot interpret {optimizer_name} for constructing an optimizer."
            logging.error(errmsg)
            raise ValueError(errmsg)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None, block=True, writer=None, writer_label="", writer_step=0):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)

    if writer is None:
        plt.show(block=block)
    else:
        writer.add_figure(writer_label, fig, writer_step)

def summarize_tensor(t, title, show_whole_tensor=False):
    if show_whole_tensor:
        print(f"{title}: {t}; Shape: {t.shape}; Type: {t.type()}")
    else:
        print(f"{title}: Shape: {t.shape}; Type: {t.type()}")