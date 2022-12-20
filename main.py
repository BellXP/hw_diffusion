import os
import argparse
import warnings
import pickle
import numpy as np
import random
import torch

from utils.config import get_config
from utils.logger import get_logger, setup_logging
from utils.run_manager import RunManager
from utils.lcp_manager import LCPManager

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_option():
    parser = argparse.ArgumentParser('Latent Code Propagation', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE")
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--model-epochs', default=100, type=int)
    parser.add_argument('--predictor-epochs', default=100, type=int)
    parser.add_argument('--train-batch-size', default=128, type=int)
    parser.add_argument('--val-batch-size', default=128, type=int)
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--use-condition', action='store_true')
    parser.add_argument('--perform-lcp', action='store_true')
    args, _ = parser.parse_known_args()

    config = get_config(args)
    return args, config


# def main_worker(gpu, ngpus_per_node, config):
def main_worker(config):
    setup_logging(os.path.join(config.out_dir, "stdout.log"))
    if config.perform_lcp:
        manager = LCPManager(config, logger)
    else:
        manager = RunManager(config, logger)
    
    # backup the config
    path = os.path.join(config.out_dir, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())

    if config.perform_lcp:
        model_names = ['mobilenet_v2', 'resnet101', 'vit_small_patch16_224']
        prop_target = 'latency#energy' # latency, energy
        arch_codes, layer_codes = manager.layer_lcp(model_names, prop_target)
        # model_lcp_results = manager.model_lcp(model_names, prop_target)
        # lcp_ckpt = {
        #     'model_names': model_names,
        #     'layer_lcp': [arch_codes, layer_codes],
        #     'model_lcp': model_lcp_results
        # }
        # with open(os.path.join(config.out_dir, "lcp_ckpt.plt"), 'wb') as f:
        #     pickle.dump(lcp_ckpt, f)
    else:
        manager.train_model()
        # manager.train_predictor()


if __name__ == '__main__':
    args, config = parse_option()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main_worker(config)
