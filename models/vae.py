import torch
import torch.nn as nn
from .modules.utils import normalization, conv_nd
from .modules.sample import Upsample, Downsample
from .modules.attention import LinAttnBlock, SpatialAttention


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def make_attn(channels, attn_type="spatial", dims=2):
    assert attn_type in ["spatial", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {channels} channels")
    if attn_type == "spatial":
        return SpatialAttention(channels, dims)
    elif attn_type == "none":
        return nn.Identity(channels)
    else:
        return LinAttnBlock(channels, dims)


class ResnetBlock(nn.Module):
    def __init__(self, channels, dropout, out_channels=None, emb_channels=None, use_conv_shortcut=False, dims=2):
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = normalization(channels)
        self.conv1 = conv_nd(
            dims, channels, out_channels, 3, stride=1, padding=1
        )
        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, out_channels)
            )
        self.norm2 = normalization(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_nd(
            dims, out_channels, out_channels, 3, stride=1, padding=1
        )
        if self.channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(
                    dims, channels, out_channels, 3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = conv_nd(
                    dims, channels, out_channels, 1, stride=1, padding=0
                )

    def forward(self, x, emb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, channels, b_channels, z_channels, resolution,
                 num_res_blocks, attn_resolutions=[], emb_channels=None,
                 b_channel_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True,
                 double_z=True, attn_type="spatial", dims=2, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(b_channel_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = conv_nd(
            dims, channels, b_channels, 3, stride=1, padding=1
        )

        curr_res = resolution
        b_channel_mult = (1,) + tuple(b_channel_mult)
        block_in = b_channels
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = b_channels * b_channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(channels=block_in,
                                         out_channels=block_out,
                                         emb_channels=emb_channels,
                                         dropout=dropout,
                                         dims=dims))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dims=dims))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, dims=dims)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=emb_channels,
                                       dropout=dropout,
                                       dims=dims)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dims=dims)
        self.mid.block_2 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=emb_channels,
                                       dropout=dropout,
                                       dims=dims)

        # end
        self.norm_out = normalization(block_in)
        self.conv_out = conv_nd(
            dims, block_in,
            2 * z_channels if double_z else z_channels,
            3, stride=1, padding=1
        )

    def forward(self, x, emb=None):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, emb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, b_channels, out_channels, z_channels, resolution,
                 num_res_blocks, attn_resolutions=[], emb_channels=None,
                 b_channel_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, give_pre_end=False, 
                 tanh_out=False, attn_type="spatial", dims=2, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(b_channel_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = b_channels * b_channel_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = conv_nd(
            dims, z_channels, block_in, 3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=emb_channels,
                                       dropout=dropout,
                                       dims=dims)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dims=dims)
        self.mid.block_2 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=emb_channels,
                                       dropout=dropout,
                                       dims=dims)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = b_channels * b_channel_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(channels=block_in,
                                         out_channels=block_out,
                                         emb_channels=emb_channels,
                                         dropout=dropout,
                                         dims=dims))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dims=dims))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, dims=dims)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = normalization(block_in)
        self.conv_out = conv_nd(
            dims, block_in, out_channels, 3, stride=1, padding=1
        )

    def forward(self, z, emb=None):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, emb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class VAE(nn.Module):
    def __init__(self, config, embed_dim, scale_factor=0.18215):
        super().__init__()
        print(f'Create VAE with scale_factor={scale_factor}')
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        self.quant_conv = conv_nd(config["dims"],
            2 * config["z_channels"] if config["double_z"] else config["z_channels"],
            2 * embed_dim, 1)
        self.post_quant_conv = conv_nd(config["dims"], embed_dim, config["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

    def encode_moments(self, x, emb=None):
        h = self.encoder(x, emb)
        moments = self.quant_conv(h)
        return moments

    def sample(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = self.scale_factor * z
        return z

    def encode(self, x, emb=None):
        moments = self.encode_moments(x, emb)
        z = self.sample(moments)
        return z

    def decode(self, z, emb=None):
        z = (1. / self.scale_factor) * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z, emb)
        return dec

    def forward(self, fn, inputs, emb=None):
        if fn == 'encode_moments':
            return self.encode_moments(inputs, emb)
        elif fn == 'encode':
            return self.encode(inputs, emb)
        elif fn == 'decode':
            return self.decode(inputs, emb)
        else:
            raise NotImplementedError


def get_vae(config=None, embed_dim=4, scale_factor=0.18215):
    if config is None:
        config = dict(
            channels=3,
            out_channels=3,
            b_channels=128,
            z_channels=4,
            resolution=256,
            double_z=True,
            b_channel_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            emb_channels=None,
            dropout=0.0,
            dims=2
        )

    return VAE(config, embed_dim, scale_factor)
