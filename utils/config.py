import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# Data settings
_C.data_root = '/nvme/xupeng/workplace/dataset/maestro_data/out-of-order_data'
_C.num_worker = 4 # the number of data 
_C.log_target = False
_C.log_layercode = False

# predictor
_C.pred_layer_dims = [256, 128]
_C.pred_act_func = 'swish'
_C.pred_use_norm = True
_C.pred_loss = 'smoothl1' # mse, l1, smoothl1

# Optimizer
_C.opt_name = 'Adam' # SGD, Adam
_C.base_lr = 1e-3
_C.min_lr = 0
_C.weight_decay = 1e-7
_C.momentum = 0.9
_C.nesterov = True

_C.print_freq = 1000
_C.save_freq = 1
_C.warmup_epochs = 5

# Prop
_C.model_defs_path = '/nvme/xupeng/workplace/lcp/data/model_defs_dict.pkl'
_C.layer_defs_path = '/nvme/xupeng/workplace/lcp/data/layer_defs.npy'
_C.prop_epoch = 100
_C.prop_step = 10
_C.prop_lr = 1


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    config.model_epochs = args.model_epochs
    config.predictor_epochs = args.predictor_epochs
    config.batch_train = args.train_batch_size
    config.batch_val = args.val_batch_size
    config.out_dir = os.path.join(args.output, args.exp_name)
    config.use_checkpoint = args.use_checkpoint
    config.use_condition = args.use_condition
    config.perform_lcp = args.perform_lcp

    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir, exist_ok=True)

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config
