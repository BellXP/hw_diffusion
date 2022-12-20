import torch
import torch.nn as nn
from .scheduler import NoiseScheduleVP
from .wrapper import model_wrapper
from .solver import DPM_Solver


def beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas


class Diffusion(nn.Module):
    def __init__(self, model, data_scale: list=[1], vae=None, loss_type='mse'):
        super().__init__()
        self.noise_schedule = NoiseScheduleVP(betas=beta_schedule())
        self.model = model
        self.vae = vae
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_scale = torch.tensor(data_scale).to(self.device)
        self.noise_steps = self.noise_schedule.total_N
        self.loss_fn = self.prepare_loss_fn(loss_type=loss_type)

    def prepare_loss_fn(self, loss_type):
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'smoothl1':
            return nn.SmoothL1Loss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)
    
    def add_noise(self, x, t, noise=None):
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        alpha_t = alpha_t[(...,) + (None,)*(x.dim() - 1)]
        sigma_t = sigma_t[(...,) + (None,)*(x.dim() - 1)]
        if noise is None:
            noise = torch.randn_like(x).to(self.device)
        x = x * alpha_t + sigma_t * noise

        return x, noise
    
    def vae_encode(self, x):
        if self.vae is None:
            return x
        return self.vae.encode(x)

    def q_sample(self, x, t=None):
        if t is None:
            t = self.sample_timesteps(x.size(0)).to(self.device)
        x = torch.div(x, self.data_scale)
        x = self.vae_encode(x)

        return self.add_noise(x, t)
    
    def p_sample(self, x, t=None, condition=None):
        model_fn = model_wrapper(self.model, self.noise_schedule, condition=condition)
        dpm_solver = DPM_Solver(model_fn, self.noise_schedule)

        if t is not None:
            t = t[0].item()
        x = dpm_solver.sample(x)
        if self.vae is not None:
            x = self.vae.decode(x)
        x = torch.mul(x, self.data_scale)

        return x

    def forward(self, x, condition=None):
        x = torch.div(x, self.data_scale)
        x_ori = x.clone()
        x = self.vae_encode(x)
        t = self.sample_timesteps(x.size(0)).to(self.device)
        x_t, noise = self.add_noise(x, t)

        model_out = self.model(x_t, t, condition)
        model_loss = self.loss_fn(model_out, noise)
        x_recon = self.p_sample(x_t, t, condition)
        x_recon = torch.div(x_recon, self.data_scale)
        recon_loss = self.loss_fn(x_recon, x_ori)
        loss = model_loss + recon_loss

        return loss, model_loss, recon_loss
