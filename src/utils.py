r""" General utilities.
"""

from logging import Formatter, StreamHandler, getLogger
import os
import random

import numpy as np
import torch


def get_logger(name="EXP"):
    r""" Generate logger.
    """

    log_fmt = Formatter(f"%(asctime)s [{name}][%(levelname)s] %(message)s ")
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed=1234):
    r""" Fix random seed for reproducibility.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
