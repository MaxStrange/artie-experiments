"""
This set of experiments attempts to cluster sounds. 
"""
from torch.utils.tensorboard import SummaryWriter
import argparse
import configuration
import datetime
import evalmetrics
import importlib
import logging
import losses
import numpy as np
import os
import pkbar
import random
import shutil
import signal
import schedulers
import subprocess
import torch
import torchinfo
import util

global_keep_training = False

def keyboard_interrupt_handler(signal, frame):
    """
    Set a global flag to stop the training.
    """
    global global_keep_training
    if global_keep_training:
        global_keep_training = False
        logging.error("CTRL+C received. Will stop after latest training iteration. Push CTRL+C again to exit without waiting.")
    else:
        exit(1)

def make_from_config_file(config: configuration.Configuration, module: str):
    """
    Dynamically load a module based on the configuration file and create the requested item from it.
    """
    experiment_type = config.getstr('Experiment', 'type')
    lib = importlib.import_module(f"{experiment_type}.{module}")
    if not hasattr(lib, "make_from_config_file"):
        logging.error(f"Cannot find 'make_from_config_file' function in module '{lib}' in package {experiment_type}")
        raise ValueError
    else:
        return lib.make_from_config_file(config)

def evaluate(network: torch.nn.Module, test, config: configuration.Configuration, device, writer: SummaryWriter, tag="Test", global_step=0):
    """
    Evaluate the neural network on the test set and print out statistics based on config file.
    """
    metrics = evalmetrics.get_metrics(config)
    network.eval()
    network.to(device)
    with torch.no_grad():
        for batch_index, (xbatch, (label_batch, freqlabel_batch)) in enumerate(test):
            xbatch = xbatch.float().to(device)
            label_batch = label_batch.to(device)

            pred_batch = network(xbatch)
            for metric in metrics:
                metric.update(pred_batch.cpu(), label_batch.cpu())

        for metric in metrics:
            metric.write_to_tensorboard(writer, tag=tag, global_step=global_step)
            if msg := metric.summarize():
                logging.info(msg)

def train_network(network: torch.nn.Module, train, val, config: configuration.Configuration, device, writer: SummaryWriter):
    """
    Train the given network and return it.
    """
    lossfn = losses.make_loss_function_from_config_file(config)
    optimizer = util.make_optimizer_from_config_file(config, network)
    scheduler, scheduler_mode = schedulers.make_scheduler_from_config_file(config, optimizer, writer)
    nepochs = config.getint('Training', 'num-epochs')
    batches_per_epoch = len(train)
    network.to(device)
    global_step = 0
    for epoch in range(nepochs):
        if not global_keep_training:
            # Received CTRL+C, we should exit
            break

        kbar = pkbar.Kbar(batches_per_epoch, epoch, nepochs)

        for batch_index, (xbatch, ybatch) in enumerate(train):
            label_batch, freqlabel_batch = ybatch
            xbatch = xbatch.float().to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            pred_batch = network(xbatch)
            loss = lossfn(pred_batch, label_batch)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_mode == "batch":
                scheduler.step(global_step)

            # Add some stats
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            util.plot_spectrogram(torch.squeeze(xbatch[0].cpu()), block=False, writer=writer, writer_label="Train/ExampleInput", writer_step=global_step)
            ## Only do these every so often because they are slooooow
            if batch_index % int(len(train) / 4) == 0:
                gradients = torch.concat([torch.flatten(p.grad.cpu()) for p in network.parameters() if p.grad is not None])
                parameters = torch.concat([torch.flatten(p.cpu()) for p in network.parameters()])
                writer.add_histogram("Train/Gradients", gradients, global_step)
                writer.add_histogram("Train/Params", parameters, global_step)

            # Update progress
            kbar.update(batch_index, values=[("loss", loss)])
            global_step += 1
        print("")  # kbar doesn't add a newline to the end of the logs
        evaluate(network, val, config, device, writer, tag="Val", global_step=global_step)
        network.train()
        if scheduler_mode == "epoch":
            scheduler.step(global_step)

    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file for this experiment.")
    parser.add_argument("-l", "--loglevel", type=str, default="info", choices=["debug", "info", "warning", "error"], help="The log level.")
    parser.add_argument("-s", "--save-directory", type=str, default="runs", help="Path to a directory to store models and history.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize a few samples from the dataset using a blocking matplotlib call.")
    parser.add_argument("--exit-after-summary", action="store_true", help="If given, we exit after we print out the summary of the network.")
    args = parser.parse_args()

    # Set up logging
    format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
    logging.basicConfig(format=format, level=getattr(logging, args.loglevel.upper()))

    # Find and load in configuration file
    if not os.path.isfile(args.config):
        logging.error(f"Cannot find {args.config} which was passed in as configuration file")
        exit(1)
    config = configuration.Configuration(args.config)

    # Create the output folder if it doesn't already exist
    save_dpath = os.path.abspath(args.save_directory)
    if os.path.exists(save_dpath) and not os.path.isdir(save_dpath):
        logging.error(f"Given a directory {args.save_directory}, but it points to {save_dpath}, which already exists and is not a directory.")
        exit(1)
    elif not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    # Create a SummaryWriter for TensorBoard
    timestamp = datetime.datetime.now().strftime("Y%Y-M%m-D%d-%H.%M")
    experiment_directory_name = config.getstr('Experiment', 'type')
    githash = subprocess.run("git log --format=%h -n 1", stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    uid = f"{timestamp}-{githash}"
    tboarddpath = os.path.join(save_dpath, experiment_directory_name, uid)
    writer = SummaryWriter(tboarddpath)

    # Create a logfile
    if not args.exit_after_summary:
        logfpath = os.path.join(tboarddpath, "logs.txt")
        filehandler = logging.FileHandler(logfpath)
        filehandler.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(filehandler)

    # Create a keyboard interrupt handler
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    # Seed all sources of randomness
    randomseed = config.getint('Experiment', 'random-seed')
    np.random.seed(randomseed)
    random.seed(randomseed)
    torch.random.manual_seed(randomseed)
    # Note that due to using a GPU and using multiprocessing, reproducibility is not guaranteed
    # But the above lines do their best

    # Get the device and log version info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    logging.info(f"Version: {torch.__version__}")

    # Create a database feeding thread based on configuration file
    logging.info("Generating dataset...")
    train, val, test = make_from_config_file(config, "dataset")

    # Get the first item from the train set so that we can look at its dims
    for spectrogram_batch, _ in train:
        dims = spectrogram_batch.shape

        if args.visualize:
            util.plot_spectrogram(torch.squeeze(spectrogram_batch[0]))
        break

    # Create a neural network based on configuration file
    logging.info("Creating neural network...")
    try:
        network = make_from_config_file(config, "neuralnetwork")
        torchinfo.summary(network, dims)
    except Exception as e:
        logging.error("Error trying to build and summarize network:", e)
    finally:
        if args.exit_after_summary:
            shutil.rmtree(tboarddpath)
            exit()

    # Train the network based on configuration file
    logging.info("Training neural network...")
    global_keep_training = True
    network = train_network(network, train, val, config, device, writer)

    # Evaluate based on configuration file
    logging.info("Evaluating neural network...")
    evaluate(network, test, config, device, writer)

    # Save based on configuration file
    # Save the git hash, a copy of the config file, the logs, and the neural network's paramaters
    logging.info(f"Saving all run information and model params to {tboarddpath}...")
    shutil.copyfile(args.config, os.path.join(tboarddpath, os.path.basename(args.config)))
    with open(os.path.join(tboarddpath, githash), 'w') as f:
        f.write(uid)
    torch.save(network.state_dict(), os.path.join(tboarddpath, "model.pth"))