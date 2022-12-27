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
    def __init__(self, model, data_scale: list=[1], vae=None, vae_emb_channel=None, loss_type='mse'):
        super().__init__()
        self.noise_schedule = NoiseScheduleVP(betas=beta_schedule())
        self.model = model
        self.vae = vae
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_scale = torch.tensor(data_scale).to(self.device)
        self.noise_steps = self.noise_schedule.total_N
        self.loss_fn = self.prepare_loss_fn(loss_type=loss_type)

        if vae is not None and vae_emb_channel is not None:
            self.vae_emb = nn.Embedding(2, vae_emb_channel).to(self.device)
        else:
            self.vae_emb = None

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
    
    def vae_encode(self, x, condition=None):
        if self.vae is None:
            return x
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)

        return self.vae.encode(x, condition)
    
    def vae_decode(self, x, condition=None):
        if self.vae is None:
            return x
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)
            
        return self.vae.decode(x, condition)

    def q_sample(self, x, t=None, condition=None):
        if t is None:
            t = torch.tensor([self.noise_steps - 1] * x.size(0)).long().to(self.device)
        x = torch.div(x, self.data_scale)
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)
        x = self.vae_encode(x, condition)

        return self.add_noise(x, t)
    
    def p_sample(self, x, t=None, condition=None):
        model_fn = model_wrapper(self.model, self.noise_schedule, condition=condition)
        dpm_solver = DPM_Solver(model_fn, self.noise_schedule)
        if t is not None:
            t = t[0].item()
        x = dpm_solver.sample(x)
        x = self.vae_decode(x, condition)
        x = x.clamp(0, 1)
        x = torch.mul(x, self.data_scale)

        return x

    def forward(self, x, condition=None, vae_recon=False):
        x = torch.div(x, self.data_scale)
        x_ori = x.clone()
        x = self.vae_encode(x, condition)
        t = self.sample_timesteps(x.size(0)).to(self.device)
        x_t, noise = self.add_noise(x, t)

        model_out = self.model(x_t, t, condition)
        model_loss = self.loss_fn(model_out, noise)

        if vae_recon:
            x_recon = self.vae_decode(x, condition)
        else:
            x_recon = self.p_sample(x_t, t, condition)
            x_recon = torch.div(x_recon, self.data_scale)
        recon_loss = self.loss_fn(x_recon, x_ori)

        return model_loss, recon_loss
