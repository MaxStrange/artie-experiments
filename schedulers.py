"""
Types of Learning Rate Scheduler.
"""
from torch.utils.tensorboard import SummaryWriter
import configuration
import logging
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
        try:
            lr = self._scheduler.get_last_lr()
        except AttributeError:
            # Nothing to do - our scheduler does not support this
            return
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

def make_scheduler_from_config_file(config: configuration.Configuration, optimizer, writer: SummaryWriter):
    """
    Returns a Scheduler and a mode str:

    "batch" => You should run the returned Scheduler after each batch
    "epoch" => You should run the returned Scheduler after each epoch
    """
    scheduler_params = config.getdict('Training', 'scheduler-params', keytype=str, valuetype=float)
    match scheduler_name := config.getstr('Training', 'scheduler'):
        case "None":
            return Scheduler(NoopLRScheduler(optimizer), writer), "epoch"
        case "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
            return Scheduler(scheduler, writer), "batch"
        case "LinearLR":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
            return Scheduler(scheduler, writer), "epoch"
        case "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
            return Scheduler(scheduler, writer), "epoch"
        case _:
            errmsg = f"Cannot interpret {scheduler_name} for constructing a learning rate scheduler."
            logging.error(errmsg)
            raise ValueError(errmsg)
