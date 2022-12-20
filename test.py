import torch
import numpy as np
from models import get_vae, UNetModel
from diffusion import Diffusion


def test_vae():
    model = get_vae()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    img = torch.randn(1, 3, 256, 256)
    img = img.to(device)
    print('test encode & decode')
    recon_img = model.decode(model.encode(img))
    print(img.shape)
    print(recon_img.shape)


    vae_config = dict(
        channels=1,
        out_channels=1,
        b_channels=32,
        z_channels=4,
        resolution=20,
        double_z=True,
        b_channel_mult=[1, 2],
        num_res_blocks=2,
        attn_resolutions=[],
        emb_channels = 12,
        dropout=0.0,
        dims=1
    )
    model = get_vae(vae_config, embed_dim=1)
    data = torch.randn(1, 1, 20)
    emb = torch.randn(1, 12)

    print('test encode & decode')
    emb_data = model.encode(data, emb)
    recon_data = model.decode(emb_data, emb)
    
    print(emb_data)
    print(recon_data)


def test_unet():
    model = UNetModel(
        image_size=20,
        in_channels=1,
        out_channels=1,
        model_channels=32,
        attention_resolutions=[],
        num_res_blocks=2,
        channel_mult=[ 1, 2],
        num_heads=1,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        use_checkpoint=False,
        legacy=False,
        num_classes=2,
        dims=1
    )
    data = torch.randn(1, 1, 20)
    emb = torch.ones(1).long()
    t = torch.ones(1)
    out = model(data, t, y=emb)

    print(out)


def test_diffusion():
    vae_config = dict(
        channels=1,
        out_channels=1,
        b_channels=32,
        z_channels=4,
        resolution=20,
        double_z=True,
        b_channel_mult=[1, 1],
        num_res_blocks=2,
        attn_resolutions=[],
        emb_channels=12,
        dropout=0.0,
        dims=1
    )
    vae = get_vae(vae_config, embed_dim=1).cuda()

    unet = UNetModel(
        image_size=20,
        in_channels=1,
        out_channels=1,
        model_channels=32,
        attention_resolutions=[],
        num_res_blocks=2,
        channel_mult=[ 1, 2],
        num_heads=1,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        use_checkpoint=False,
        legacy=False,
        dims=1,
        num_classes=2
    ).cuda()

    diffusion = Diffusion(unet, data_scale=[1], vae=None).cuda()

    data = torch.randn(1, 1, 20).cuda()
    label = torch.ones(1).long().cuda()

    t = diffusion.sample_timesteps(1)
    x, noise = diffusion.q_sample(data, t)
    data = diffusion.p_sample(x, t, label)

    data = torch.randn(1, 1, 20).cuda()
    loss, model_loss, recon_loss = diffusion(data, label)
    print(loss)
    print(model_loss)
    print(recon_loss)


if __name__ == "__main__":
    # test_vae()
    # test_unet()
    test_diffusion()