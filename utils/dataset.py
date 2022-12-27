import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import linalg
import pickle


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


class CodeDataset(Dataset):
    def __init__(self, config):
        # get data
        ckpt = pickle.load(open('/nvme/xupeng/workplace/dataset/linear_try/linear_try_data.pkl', 'rb'))
        arch_code = ckpt['legal_arch_codes']
        illegal_arch_code = ckpt['illegal_arch_codes']

        config.defrost()
        config.arch_mu = np.mean(arch_code, axis=0).tolist()
        config.arch_sigma = np.cov(arch_code, rowvar=False).tolist()
        config.freeze()

        self.dataset = torch.tensor(np.concatenate([arch_code, illegal_arch_code], axis=1)).type(torch.float)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        arch_code, illegal_arch_code = torch.chunk(data_sample, 2, dim=0)

        return arch_code, illegal_arch_code
    

class PredictorDataset(Dataset):
    def __init__(self, arch_codes, latent_codes):
        # get data
        ckpt = pickle.load(open('/nvme/xupeng/workplace/dataset/linear_try/linear_try_data.pkl', 'rb'))
        layer_codes = ckpt['layer_codes']
        energy_line_kernel = ckpt['energy_line_kernel']
        runtime_line_kernel = ckpt['runtime_line_kernel']
        arch_codes = arch_codes.detach().cpu().numpy()
        latent_codes = latent_codes.reshape(len(arch_codes), -1).repeat(len(layer_codes), 1).cpu().numpy()

        data_input = []
        for arch_code in arch_codes:
            for layer_code in layer_codes:
                input_code = np.concatenate([layer_code, arch_code])
                data_input.append(input_code)
        data_input = np.array(data_input)
        data_layer = data_input[:, :7]
        data_energy = (data_input * energy_line_kernel).sum(axis=1).reshape(-1, 1)
        data_runtime = (data_input * runtime_line_kernel).sum(axis=1).reshape(-1, 1)

        self.dataset = torch.tensor(np.concatenate([data_layer, latent_codes, data_runtime, data_energy], axis=1)).type(torch.float)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_sample = self.dataset[idx]
        layer_code = data_sample[:7]
        latent_code = data_sample[7:-2]
        runtime = data_sample[-2]
        energy = data_sample[-1]

        return layer_code, latent_code, runtime, energy


def get_dataloader(config):
    code_dataset = CodeDataset(config)
    whole_length = len(code_dataset)
    train_length = int(0.8 * whole_length)
    val_length = whole_length - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(code_dataset, [train_length, val_length])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_train,
                                  shuffle=True,
                                  num_workers=config.num_worker)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config.batch_val,
                                shuffle=False,
                                num_workers=config.num_worker)

    return train_dataloader, val_dataloader
