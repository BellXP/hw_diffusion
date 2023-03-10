import torch
import torch.nn as nn
import copy
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
        self.noise_schedule = NoiseScheduleVP(betas=beta_schedule(n_timestep=200))
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
            
        self.q_model = copy.deepcopy(model)

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
        x_t = x * alpha_t + sigma_t * noise

        return x_t, noise
    
    def vae_encode(self, x, condition=None):
        if self.vae is None:
            return x, 0
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)
        moments = self.vae.encode_moments(x, condition)
        x = self.vae.sample(moments)

        mean, logvar = torch.chunk(moments, 2, dim=1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1))

        return x, kld_loss
    
    def vae_decode(self, x, condition=None):
        if self.vae is None:
            return x
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)
            
        return self.vae.decode(x, condition)

    def q_sample(self, x, t=None, condition=None, use_q_model=False):
        if t is None:
            t = torch.tensor([self.noise_steps - 1] * x.size(0)).long().to(self.device)
        x = torch.div(x, self.data_scale)
        if condition is not None and self.vae_emb is not None:
            condition = self.vae_emb(condition)
        x, kld_loss = self.vae_encode(x, condition)
        
        if use_q_model:
            alpha_t = self.noise_schedule.marginal_alpha(t)
            sigma_t = self.noise_schedule.marginal_std(t)
            alpha_t = alpha_t[(...,) + (None,)*(x.dim() - 1)]
            sigma_t = sigma_t[(...,) + (None,)*(x.dim() - 1)]
            x_t = self.q_model(x, t, condition)
            noise = torch.div(x_t - x * alpha_t, sigma_t)
        else:
            x_t, noise = self.add_noise(x, t)
        
        return x_t, noise
    
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
        x, kld_loss = self.vae_encode(x, condition)
        t = self.sample_timesteps(x.size(0)).to(self.device)
        x_t, noise = self.add_noise(x, t)

        model_out = self.model(x_t, t, condition)
        model_loss = self.loss_fn(model_out, noise)
        
        q_model_out = self.q_model(x, t, condition)
        q_model_loss = self.loss_fn(q_model_out, x_t)
        
        model_loss = model_loss + q_model_loss
        
        if vae_recon:
            x_recon = self.vae_decode(x, condition)
        else:
            x_recon = self.p_sample(x_t, t, condition)
            x_recon = torch.div(x_recon, self.data_scale)
        recon_loss = self.loss_fn(x_recon, x_ori)
        
        return model_loss, recon_loss, kld_loss
