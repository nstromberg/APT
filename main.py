import os
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from apt.train import train

def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser()

    # setting arguments
    parser.add_argument("--name", default="default_run")
    parser.add_argument("--artifact_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", default="cuda:0")

    # model arguments
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=4)

    # data arguments
    parser.add_argument("--data_size", type=int, default=1000)
    parser.add_argument("--reset_freq", type=int, default=5)
    parser.add_argument("--num_datasets", type=int, default=8)
    parser.add_argument("--num_trained_datasets", type=int, default=2)

    # training arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--classification", action="store_true")
    parser.add_argument("--state_dict", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--data_weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--aggregate_k_gradients", type=int, default=2)
    parser.add_argument("--initial_eval", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--checkpoint_freq", type=int, default=2)

    return parser.parse_args()

def initialize_logger(artifact_path, name=None, level='INFO'):
    logfile = os.path.join(artifact_path, 'log.txt')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    handler_console = logging.StreamHandler()
    handler_file    = logging.FileHandler(logfile)

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger


if __name__ == "__main__":
    args = parse_arguments()
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, "runs/" + args.name + "_" + dt))
    save_dir = os.path.join(args.artifact_path, "saves/" + args.name + "_" + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)
    logging.info(vars(args))

    train(args, writer, save_dir)
