from clustering.networks import autoencoder
from clustering.networks import classifier
import logging

def make_from_config_file(config):
    """
    Create the neural network from the given configuration file.
    """
    match nettype := config.getstr('Network', 'type').lower():
        case 'classifier':
            return classifier.make_from_config_file(config)
        case 'autoencoder':
            return autoencoder.make_from_config_file(config)
        case _:
            errmsg = f"Do not recognize neural network type {nettype}. Please implement it or change the value in the configuration file."
            logging.error(errmsg)
            raise ValueError(errmsg)