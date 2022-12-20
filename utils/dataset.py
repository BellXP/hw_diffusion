import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from math import ceil


class DCPDataset(Dataset):
    def __init__(self, config):
        # get data
        data_input = np.load(f'{config.data_root}/filtered_input.npy')
        layer_code = data_input[: , :7]
        arch_code = data_input[:, 7:]
        data_output = np.load(f'{config.data_root}/filtered_input.npy')[:, [0, 2]] # only use runtime and energy

        # process data
        data = {}
        if config.log_target:
            data_output = np.log(data_output)
        if config.log_layercode:
            layer_code = np.log(layer_code)
        self.dataset = torch.tensor(np.concatenate([layer_code, arch_code, data_output], axis=1)).to(torch.float32)

        json_path = os.path.join(config.out_dir, f'dcp_dataset_stats.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        config.defrost()
        config.condition_dim = 7
        config.latent_dim = arch_code.shape[1]
        config.freeze()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        layer_code = data_sample[:7]
        arch_code = data_sample[7:-2]
        runtime = data_sample[-2]
        energy = data_sample[-1]

        return arch_code, runtime, energy, layer_code


class TryDataset(Dataset):
    def __init__(self, config):
        # get data
        import pickle
        ckpt = pickle.load(open('/nvme/xupeng/workplace/dataset/linear_try/linear_try_data.pkl', 'rb'))
        data_input = ckpt['data_input']
        data_energy = ckpt['data_energy'].reshape(-1, 1)
        data_runtime = ckpt['data_runtime'].reshape(-1, 1)
        layer_code = data_input[:, :7]
        arch_code = data_input[:, 7:]

        # process data
        if config.log_target:
            data_energy = np.log(data_runtime)
            data_runtime = np.log(data_runtime)
        if config.log_layercode:
            layer_code = np.log(layer_code)
        self.dataset = torch.tensor(np.concatenate([layer_code, arch_code, data_energy, data_runtime], axis=1)).type(torch.float)

        # illegal_data
        illegal_arch_code = ckpt['illegal_arch_codes']
        unique_layer_code_num = 1000
        self.illegal_arch_code = torch.tensor(illegal_arch_code).type(torch.float).repeat((unique_layer_code_num, 1))

        config.defrost()
        config.condition_dim = 7
        config.latent_dim = arch_code.shape[1]
        config.freeze()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        layer_code = data_sample[:7]
        arch_code = data_sample[7:-2]
        runtime = data_sample[-1]
        energy = data_sample[-2]

        illegal_arch_code = self.illegal_arch_code[idx]

        return arch_code, illegal_arch_code, runtime, energy, layer_code


def get_dataloader(config):
    pred_dataset = TryDataset(config)
    whole_length = len(pred_dataset)
    train_length = int(0.8 * whole_length)
    val_length = whole_length - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(pred_dataset, [train_length, val_length])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_train,
                                  shuffle=True,
                                  num_workers=config.num_worker)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config.batch_val,
                                shuffle=False,
                                num_workers=config.num_worker)

    return train_dataloader, val_dataloader
