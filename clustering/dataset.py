"""
Dataset code specific to clustering experiments.
"""
from clustering.datasets import autoencoder
from clustering.datasets import classifier
import logging

def make_from_config_file(config):
    """
    Make a clustering experiment's dataset based on the given configuration file.
    """
    match dataset_type := config.getstr('Dataset', 'type'):
        case "autoencoder":
            return autoencoder.make_from_config_file(config)
        case "classifier":
            return classifier.make_from_config_file(config)
        case _:
            errmsg = f"Cannot interpret {dataset_type} from configuration file for Dataset type."
            logging.error(errmsg)
            raise ValueError(errmsg)