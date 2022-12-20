import torch
import torch.nn as nn
from .modules.activations import build_activation
import math


class MLP(nn.Module):
    def __init__(self, hidden_features, in_features, out_features, bias=True, act_func=None):
        super(MLP, self).__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features
        self.bias = bias
        self.act_func = act_func

        self.linear1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_features, bias=self.bias)
        # self.norm = nn.BatchNorm1d(self.hidden_features)
        self.drop = nn.Dropout(0.1)
        self.act = build_activation(self.act_func, inplace=True)
        self.linear2 = nn.Linear(in_features=self.hidden_features, out_features=self.out_features, bias=self.bias)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.norm(x)
        x = self.drop(x)
        if self.act is not None:
            x = self.act(x)
        x = self.linear2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, hidden_features, out_features, act_func='swish', qkv_bias=True):
        super(Attention, self).__init__()
        self.token_num = 8
        self.dim = dim // self.token_num
        self.hidden_features = hidden_features // self.token_num
        self.out_features = out_features // self.token_num
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        
        assert dim % self.token_num == 0, "Attention in feature cannot be divided by token num"
        assert hidden_features % self.token_num == 0, "Attention hidden feature cannot be divided by token num"
        assert out_features % self.token_num == 0, "Attention out feature cannot be divided by token num"
        assert self.dim % self.num_heads == 0, "Attention dim cannot be divided by head num"
        assert self.hidden_features % self.num_heads == 0, "Attention hidden feature cannot be divided by head num"

        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.q = nn.Linear(in_features=self.dim, out_features=self.dim, bias=qkv_bias)
        self.k = nn.Linear(in_features=self.dim, out_features=self.dim, bias=qkv_bias)
        self.v = nn.Linear(in_features=self.dim, out_features=self.hidden_features, bias=qkv_bias)

        self.norm = nn.LayerNorm(self.hidden_features)
        self.proj = nn.Linear(in_features=self.hidden_features, out_features=self.out_features)
        self.proj_l = nn.Linear(in_features=self.num_heads, out_features=self.num_heads)
        self.proj_w = nn.Linear(in_features=self.num_heads, out_features=self.num_heads)
        
        self.softmax = nn.Softmax(dim=-1)
        self.act_func = act_func
        self.act = build_activation(act_func, inplace=True)

        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B_, C = x.shape
        N = self.token_num
        x = x.reshape(B_, N, -1)

        q = self.q(x).reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.softmax(attn)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        v = self.v(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.norm(x)
        x = self.proj(self.act(x))
        x = self.proj_drop(x).reshape(B_, -1)

        return x


class Predictor(nn.Module):
    def __init__(self, latent_dim, condition_dim, layer_dims: list, act_func='tanh', use_norm=False):
        super(Predictor, self).__init__()
        self.act_func = build_activation(act_func, inplace=True)
        self.norm_func = nn.BatchNorm1d if use_norm else nn.Identity
        self.flatten = nn.Flatten()

        layer_dims = [256, 128]
        self.in_func = nn.Sequential(
            self.norm_func(latent_dim + condition_dim),
            nn.Linear(latent_dim + condition_dim, layer_dims[0], bias=True),
            self.norm_func(layer_dims[0]),
            self.act_func,
            nn.Dropout(0.5),
        )

        # self.encoder = nn.Sequential()
        # for i in range(len(layer_dims) -1):
        #     self.encoder.add_module(f'up_norm_{i}', self.norm_func(layer_dims[i]))
        #     # self.encoder.add_module(f'up_attn_{i}', Attention(layer_dims[i], 1, layer_dims[i], layer_dims[i+1], act_func))
        #     self.encoder.add_module(f'up_mlp_{i}', MLP(layer_dims[i] * 2, layer_dims[i], layer_dims[i+1], True, act_func))
        #     self.encoder.add_module(f'up_act_{i}', self.act_func)
        self.out_runtime = nn.Sequential(
            nn.Linear(layer_dims[-2], layer_dims[-1]),
            nn.BatchNorm1d(layer_dims[-1]),
            self.act_func,
            nn.Dropout(0.5),
            nn.Linear(layer_dims[-1], 1)
        )

        self.out_energy = nn.Sequential(
            nn.Linear(layer_dims[-2], layer_dims[-1]),
            nn.BatchNorm1d(layer_dims[-1]),
            self.act_func,
            nn.Dropout(0.5),
            nn.Linear(layer_dims[-1], 1)
        )

    def forward(self, latent_code, condition_code):
        latent_code = self.flatten(latent_code)
        condition_code = self.flatten(condition_code)
        pred_input = torch.cat((latent_code, condition_code), dim=1)
        x = self.in_func(pred_input)
        # x = self.encoder(x)
        r_pred = self.out_runtime(x)
        e_pred = self.out_energy(x)

        return r_pred, e_pred

    def init_model(self):
        """ Linear, BatchNorm1d, BatchNorm2d """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_predictor(config):
    predictor = Predictor(config.latent_dim * 2, config.condition_dim, config.pred_layer_dims, config.pred_act_func, config.pred_use_norm)
    predictor.init_model()

    return predictor


def get_criterion(pred_loss):
    if pred_loss == 'mse':
        return nn.MSELoss()
    elif pred_loss == 'l1':
        return nn.L1Loss()
    elif pred_loss == 'smoothl1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError('Invalid Pred loss type')
    