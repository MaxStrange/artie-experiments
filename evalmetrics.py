"""
A bunch of metrics to use for evaluation in various experiments.
"""
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from typing import List
import configuration
import librosa
import logging
import losses
import matplotlib.pyplot as plt
import torch

class Metric:
    """
    Base class for all metrics. They should implement these functions.
    """
    def __init__(self):
        pass

    def update(self, predicted_batch, label_batch):
        # Should update the internal state with this batch's performance
        pass

    def summarize(self) -> str:
        # Should return whatever you want to show to the user in str form
        pass

    def write_to_tensorboard(self, writer: SummaryWriter, tag="Test", global_step=0):
        # Should write something useful to the given tensorboard
        pass

class AccuracyMetric(Metric):
    """
    Metric for calculating the accuracy of a classification network.
    """
    def __init__(self, config: configuration.Configuration):
        super().__init__()
        self._correct = 0
        self._nguesses = 0

    def update(self, predicted_batch, label_batch):
        self._correct += (predicted_batch.argmax(1) == label_batch).type(torch.float).sum().item()
        self._nguesses += predicted_batch.shape[0]  # batch size

    def summarize(self) -> str:
        return f"Accuracy: {(100 * (self._correct / self._nguesses)):>0.1f}%"

    def write_to_tensorboard(self, writer: SummaryWriter, tag="Test", global_step=0):
        writer.add_text(f"{tag}/Accuracy", f"{(100 * (self._correct / self._nguesses)):>0.1f}%", global_step=global_step)

class ConfusionMatrixMetric(Metric):
    """
    Metric for calculating a confusion matrix.
    """
    def __init__(self, config: configuration.Configuration):
        super().__init__()
        self._predicted_class_indexes = []
        self._ground_truth_class_indexes = []

    def update(self, predicted_batch, label_batch):
        predicted_classes = predicted_batch.argmax(1)
        assert len(predicted_classes) == len(label_batch)
        for predicted_class, label in zip(predicted_classes, label_batch):
            self._predicted_class_indexes.append(predicted_class)
            self._ground_truth_class_indexes.append(label)

    def summarize(self) -> str:
        return f"Confusion Matrix:\n{confusion_matrix(self._ground_truth_class_indexes, self._predicted_class_indexes)}"

    def write_to_tensorboard(self, writer: SummaryWriter, tag="Test", global_step=0):
        cm = confusion_matrix(self._ground_truth_class_indexes, self._predicted_class_indexes)
        cmdisplay = ConfusionMatrixDisplay(cm).plot()
        writer.add_figure(f"{tag}/ConfusionMatrix", cmdisplay.figure_, global_step=global_step)

class LossMetric(Metric):
    """
    Metric for calculating the network's training loss function.
    """
    def __init__(self, config: configuration.Configuration):
        super().__init__()
        self._loss_function = losses.make_loss_function_from_config_file(config)
        self._loss = 0.0
        self._nbatches = 0

    def update(self, predicted_batch, label_batch):
        self._loss += self._loss_function(predicted_batch, label_batch).item()
        self._nbatches += 1

    def summarize(self) -> str:
        return f"Average Loss: {(self._loss / self._nbatches):>8f}"

    def write_to_tensorboard(self, writer: SummaryWriter, tag="Test", global_step=0):
        writer.add_text(f"{tag}/Loss", f"{(self._loss / self._nbatches):>8f}", global_step=global_step)

class ReconstructionImageMetric(Metric):
    """
    Not really a metric, just a way to update Tensorboard with predicted spectrograms.
    """
    def __init__(self, config):
        super().__init__()
        self._example_inputs = None
        self._example_labels = None
        self._batchsize = config.getint('Dataset', 'batch-size')

    def update(self, predicted_batch, label_batch):
        # Grab and store the first few examples, then we'll display them later
        if self._example_inputs is None:
            self._example_inputs = predicted_batch[0:min(4, self._batchsize)]
            self._example_labels = label_batch[0:min(4, self._batchsize)]

    def summarize(self) -> str:
        # Nothing to summarize - we exist solely to write to Tensorboard
        return None

    def write_to_tensorboard(self, writer: SummaryWriter, tag="Test", global_step=0):
        for i in range(self._example_inputs.shape[0]):
            figure, axs = plt.subplots(1, 2)
            # Create two spectrograms on a single plot: label and predicted
            spec0 = torch.squeeze(self._example_labels[i].cpu())
            img0 = axs[0].imshow(librosa.power_to_db(spec0), origin="lower", aspect="auto")
            figure.colorbar(img0, ax=axs[0])
            # Predicted
            spec1 = torch.squeeze(self._example_inputs[i].cpu())
            img1 = axs[1].imshow(librosa.power_to_db(spec1), origin="lower", aspect="auto")
            figure.colorbar(img0, ax=axs[1])

            writer.add_figure(f"{tag}/ReconstructedSpectrogram_{i}", figure, global_step=global_step)


def get_metrics(config: configuration.Configuration) -> List[Metric]:
    metric_names = config.getlist('Evaluation', 'metrics', type=str)
    metrics = []
    for name in metric_names:
        match name.lower():
            case "accuracy":
                metrics.append(AccuracyMetric(config))
            case "loss":
                metrics.append(LossMetric(config))
            case "confusion-matrix":
                metrics.append(ConfusionMatrixMetric(config))
            case "reconstruction-image":
                metrics.append(ReconstructionImageMetric(config))
            case _:
                errmsg = f"Cannot interpret metric named {name} when trying to add metrics to evaluation task"
                logging.error(errmsg)
                raise ValueError(errmsg)
    return metrics