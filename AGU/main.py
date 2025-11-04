import logging
import os
import sys
import random
import numpy as np

import torch
from exp_agu import ExpAGU
from parameter_parser import parameter_parser
import warnings

warnings.filterwarnings("ignore")

def config_logger():
    logging.getLogger("root").setLevel(logging.ERROR)

    logger = logging.getLogger("ExpAGU")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def _set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parameter_parser()
    _set_random_seed(42)
    config_logger()

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Exp = ExpAGU(args)