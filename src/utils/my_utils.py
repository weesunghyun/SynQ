import os
import sys
import random
import logging
import warnings
from datetime import datetime

import numpy as np

import torch
from torch import nn

def fix_seed(seed):
    """
    Fix the seed for reproducibility
    Args:
        seed: seed number
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

def check_path(model_path):
    """
    Check the path of model if it exists, if not, create it
    Args:
        model_path: path of model
    """

    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def to_device(gpu):
    """
    Return the device
    Args:
        gpu: gpu id
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def print_tensors_shape(l):
    """
    Print the shape of tensors
    Args:
        l: list of tensors
    """

    print(f">> {len(l)} tensors")
    for item in l:
        print(f"  {item.shape}")


def filter_warnings():
    """
    Filter warnings
    """

    warnings.filterwarnings("ignore")


def setup_logging_with_path(log_path):
    """
    Setup the logging
    Args:
        log_path: path of log file
    """

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_path,
                    filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s: %(levelname)-6s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    class StreamToLogger(object):
        """
        Fake file-like stream object that redirects writes to a logger instance.
        """
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''

        def write(self, buf):
            """
            Write the buffer
            Args:
                buf: buffer
            """

            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        # def flush(self):
        #     pass

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)


def setup_logging(filename):
    """
    Setup the logging
    Args:
        filename: name of log file
    """

    log_path = f"logs/{filename}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    check_path(log_path)
    setup_logging_with_path(log_path)


def basic_setup(filename):
    """
    Basic setup
    Args:
        filename: name of log file
    """

    fix_seed(0)
    filter_warnings()
    setup_logging(filename)


def freeze_model(self,model):
    """
    freeze the activation range
    Args:
        model: model to freeze
    """

    if isinstance(model, torch.nn.Sequential):
        for _, m in model.named_children():
            self.freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                self.freeze_model(mod)
        return model


def unfreeze_model(self,model):
    """
    unfreeze the activation range
    Args:
        model: model to unfreeze
    """

    if isinstance(model, nn.Sequential):
        for _, m in model.named_children():
            self.unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, torch.nn.Module) and 'norm' not in attr:
                self.unfreeze_model(mod)
        return model
