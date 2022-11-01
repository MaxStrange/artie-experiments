"""
Custom losses for the different experiments.
"""
from torch import nn
import configuration
import logging
import torch

def reconstruction_loss(pred_batch, label_batch):
    """
    A loss function that balances the 0-valued pixels against the non-zero
    valued pixels in the spectrograms.
    """
    mask_zeros_in_label = torch.isclose(label_batch, torch.zeros_like(label_batch)).int()
    batch_sums_of_num_zeros_in_label = torch.sum(mask_zeros_in_label, (1, 2, 3), dtype=label_batch.dtype)
    mask_nonzeros_in_label = (label_batch > 0.0).int()
    batch_sums_of_num_non_zeros_in_label = torch.sum(mask_nonzeros_in_label, (1, 2, 3), dtype=label_batch.dtype)
    batch_balance = batch_sums_of_num_zeros_in_label / batch_sums_of_num_non_zeros_in_label
    balance = torch.mean(batch_balance, dtype=label_batch.dtype)
    balances = torch.ones_like(label_batch, dtype=label_batch.dtype)
    balances[mask_nonzeros_in_label.bool()] = balance
    return torch.mean(torch.abs(pred_batch - label_batch) * balances)

def make_loss_function_from_config_file(config: configuration.Configuration):
    """
    """
    match lossname := config.getstr('Training', 'loss-function'):
        case "NLLLoss":
            return torch.nn.NLLLoss()
        case "L1Loss":
            return torch.nn.L1Loss()
        case "ReconstructionLoss":
            return reconstruction_loss
        case _:
            errmsg = f"Cannot interpret {lossname} for constructing a loss function."
            logging.error(errmsg)
            raise ValueError(errmsg)

if __name__ == "__main__":
    pred_batch = torch.rand((2, 1, 10, 5), dtype=torch.float32, requires_grad=True)
    label_batch = torch.zeros_like(pred_batch, dtype=torch.float32)
    label_batch[:, :, 4, :] = 1.0
    loss = reconstruction_loss(pred_batch, label_batch)
    print(loss, torch.nn.functional.l1_loss(pred_batch, label_batch))