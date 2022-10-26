"""
This set of experiments attempts to cluster sounds. 
"""
import argparse
import configuration
import importlib
import logging
import numpy as np
import os
import random
import torch
import torchaudio
import util

def make_dataset_from_config_file(config: configuration.Configuration):
    """
    Create a training and testing DataLoader from the configuration file.
    """
    experiment_type = config.getstr('Experiment', 'type')
    datalib = importlib.import_module(f"{experiment_type}.dataset")
    if not hasattr(datalib, "make_dataset_from_config_file"):
        logging.error(f"Cannot find 'make_dataset_from_config_file' function in module 'dataset.py' in package {experiment_type}")
        raise ValueError("Cannot make dataset with given configuration file")
    else:
        return datalib.make_dataset_from_config_file(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file for this experiment.")
    parser.add_argument("-l", "--loglevel", type=str, default="info", choices=["debug", "info", "warning", "error"], help="The log level.")
    args = parser.parse_args()

    # Set up logging
    format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
    logging.basicConfig(format=format, level=getattr(logging, args.loglevel.upper()))

    # Find and load in configuration file
    if not os.path.isfile(args.config):
        logging.error(f"Cannot find {args.config} which was passed in as configuration file")
        exit(1)
    config = configuration.Configuration(args.config)

    # Seed all sources of randomness
    randomseed = config.getint('Experiment', 'random-seed')
    np.random.seed(randomseed)
    random.seed(randomseed)
    torch.random.manual_seed(randomseed)
    # Note that due to using a GPU and using multiprocessing, reproducibility is not guaranteed
    # But the above lines do their best

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    # Create a database feeding thread based on configuration file
    train, test = make_dataset_from_config_file(config)

    ##########################################################
    ################# Visualize the dataset ##################
    ##########################################################
    # Take a look at the first N items
    for i, (spectrogram, freqlabel) in enumerate(train):
        util.plot_waveform(spectrogram, config.getfloat('Dataset', 'sample-rate-hz'))
        util.plot_specgram(spectrogram, config.getfloat('Dataset', 'sample-rate-hz'))
        util.print_stats(spectrogram, config.getfloat('Dataset', 'sample-rate-hz'))
        if i >= 2:
            break
    ##########################################################

    # Create a neural network based on configuration file

    # Train the network based on configuration file

    # Evaluate based on configuration file

    # Save based on configuration file