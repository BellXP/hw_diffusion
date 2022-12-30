import torch
from timm.utils import AverageMeter
import time
import os
from datetime import timedelta
from math import *
import numpy as np

from utils.dataset import get_dataloader, calculate_frechet_distance, PredictorDataset, DataLoader
from models.predictor import build_predictor, get_criterion
from models import get_vae, UNetModel
from diffusion import Diffusion


def satisfy_arch_constraint(arch_code):
    arch_code = arch_code.reshape(-1, 5, 4)
    for i in range(arch_code.shape[1] - 1):
        if arch_code[:, i, :].sum() < arch_code[:, i+1, :].sum():
            return False
    return True


def biasd_accuracy(output, target):
    accuracy = 1 - (abs(output - target) / abs(target))
    accuracy = accuracy.sum(dim=0) / accuracy.size(0)
    
    return accuracy.item()


class RunManager:
    def __init__(self, config, logger, load_best=False):
        self.config = config
        self.logger = logger
        self._save_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get model
        vae = get_vae(dict(
            channels=1,
            out_channels=1,
            b_channels=32,
            z_channels=4,
            resolution=20,
            double_z=False,
            b_channel_mult=[1, 2],
            num_res_blocks=2,
            attn_resolutions=[20, 10],
            emb_channels=32,
            dropout=0.0,
            dims=1
        ), embed_dim=4).to(self.device)
        unet = UNetModel(
            image_size=10,
            in_channels=4,
            out_channels=4,
            model_channels=32,
            attention_resolutions=[],
            num_res_blocks=2,
            channel_mult=[1, 2],
            num_heads=1,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            use_checkpoint=False,
            legacy=False,
            dims=1,
            num_classes=2
        ).to(self.device)
        data_scale = [10, 20, 30, 40] * 5
        self.model = Diffusion(unet, data_scale=data_scale, vae=vae if config.use_vae else None, vae_emb_channel=32).to(self.device)
        self.opt_model = self.build_optimizer(self.model.parameters())
        self.kld_weight = 1.0

        # get predictor
        self.predictor = build_predictor(config).to(self.device)
        self.predictor_criterion = get_criterion(config.pred_loss)
        self.opt_predictor = self.build_optimizer(self.predictor.parameters())

        # training
        self.vae_warmup = True if self.config.use_vae else False
        self.model_epoch, self.predictor_epoch = 0, 0
        self.model_loss, self.predictor_loss = float('inf'), float('inf')
        self._num_epochs, self._num_steps = None, None
        if config.use_checkpoint:
            self.load_checkpoint(load_best)

    def build_optimizer(self, params):
        if self.config.opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.base_lr
            )
        else:
            raise ValueError('Invalid type of Optimizer')

        return optimizer

    def _calc_learning_rate(self, epoch, idx):
        if epoch < self.config.warmup_epochs:
            lr = self.config.min_lr + (self.config.base_lr - self.config.min_lr) * (epoch * self._num_steps + idx) / (self.config.warmup_epochs * self._num_steps)
        else:
            T_max = self._num_steps * (self._num_epochs - self.config.warmup_epochs)
            T_cur = self._num_steps * (epoch - self.config.warmup_epochs) + idx
            # cosine schedule
            # lr = self.config.min_lr + 0.5 * (self.config.base_lr - self.config.min_lr) * (1 + cos(pi * T_cur / T_max))
            # linear schedule
            lr = self.config.min_lr + (self.config.base_lr - self.config.min_lr) * (1 - T_cur / T_max)
            
        return lr

    def adjust_learning_rate(self, optimizer, epoch, idx):
        lr = self._calc_learning_rate(epoch, idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.config.out_dir, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'vae_warmup': self.vae_warmup,
            'model_epoch': self.model_epoch,
            'model_loss': self.model_loss,
            'model': self.model.state_dict(),
            'opt_model': self.opt_model.state_dict(),
            'predictor_epoch': self.predictor_epoch,
            'predictor_loss': self.predictor_loss,
            'opt_predictor': self.opt_predictor.state_dict(),
            'predictor': self.predictor.state_dict()
        }

        model_path = os.path.join(self.save_path, 'ckpt.pth')
        self.logger.info(f"{model_path} saving......")
        with open(model_path, 'wb') as f:
            torch.save(checkpoint, f)
        self.logger.info(f"{model_path} saved !!!")

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth')
            with open(best_path, 'wb') as f:
                torch.save(checkpoint, f)
            self.logger.info(f"{best_path} saved !!!")

    def load_checkpoint(self, load_best=False):
        if load_best:
            model_path = os.path.join(self.save_path, 'model_best.pth')
        else:
            model_path = os.path.join(self.save_path, 'ckpt.pth')

        if not os.path.exists(model_path):
            self.logger.info('fail to load checkpoint from %s' % model_path)
            return

        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')

        self.vae_warmup = checkpoint['vae_warmup']
        self.model_epoch = checkpoint['model_epoch'] + 1
        self.model_loss = checkpoint['model_loss']
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        self.logger.info(msg)
        self.predictor_epoch = checkpoint['predictor_epoch'] + 1
        self.predictor_loss = checkpoint['predictor_loss']
        self.opt_predictor.load_state_dict(checkpoint['opt_predictor'])
        msg = self.predictor.load_state_dict(checkpoint['predictor'], strict=False)
        self.logger.info(msg)
        self.logger.info('load checkpoint from %s' % model_path)

    @torch.no_grad()
    def validate_model(self, sample_num=1000):
        self.model.eval()
        num_steps = len(self.val_dataloader)
        model_loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        kld_loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        fid_meter = AverageMeter()

        for idx, [arch_code, illegal_arch_code] in enumerate(self.val_dataloader):
            arch_code = arch_code.to(self.device)
            if self.config.use_condition:
                illegal_arch_code = illegal_arch_code.to(self.device)
                x_cond = torch.zeros(arch_code.size(0) * 2).to(torch.int32).to(self.device)
                x_cond[arch_code.size(0):] = 1
                arch_code = torch.concatenate([arch_code, illegal_arch_code], dim=0)
            else:
                x_cond = None

            arch_code = arch_code.unsqueeze(dim=1)
            model_loss, recon_loss, kld_loss = self.model(arch_code, condition=x_cond, vae_recon=self.vae_warmup)

            arch_codes, _ = self.random_gen_arch_codes(sample_num)
            correct_num = sum([int(satisfy_arch_constraint(arch_code)) for arch_code in arch_codes])
            acc = correct_num * 100.0 / sample_num

            arch_codes = arch_codes.cpu().numpy()
            fid_score = calculate_frechet_distance(
                np.array(self.config.arch_mu),
                np.array(self.config.arch_sigma),
                np.mean(arch_codes, axis=0),
                np.cov(arch_codes, rowvar=False)
            )

            fid_meter.update(fid_score)
            acc_meter.update(acc)
            model_loss_meter.update(model_loss.item(), arch_code.size(0))
            recon_loss_meter.update(recon_loss.item(), arch_code.size(0))
            kld_loss_meter.update(kld_loss.item(), arch_code.size(0))
        
            if idx % self.config.print_freq == 0 or idx + 1 == num_steps:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                model_name = 'VAE' if self.vae_warmup else 'Diffusion'
                batch_log = f'{model_name} Valid: [{self.model_epoch + 1}/{self.config.model_epochs}][{idx}/{num_steps}]\t' \
                        f'model loss {model_loss_meter.val:.4f} ({model_loss_meter.avg:.4f})\t' \
                        f'recon loss {recon_loss_meter.val:.4f} ({recon_loss_meter.avg:.4f})\t' \
                        f'kld loss {kld_loss_meter.val:.4f} ({kld_loss_meter.avg:.4f})\t' \
                        f'acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})\t' \
                        f'FID {fid_meter.val:.4f} ({fid_meter.avg:.4f})\t' \
                        f'mem {memory_used:.0f}MB'
                self.logger.info(batch_log)
        
        return model_loss_meter.avg + recon_loss_meter.avg + self.kld_weight * kld_loss_meter.avg

    def train_model(self):
        self.model.train()
        self.train_dataloader, self.val_dataloader = get_dataloader(self.config)
        num_steps = len(self.train_dataloader)
        self._num_steps = num_steps
        self._num_epochs = self.config.model_epochs

        while self.model_epoch < self.config.model_epochs:
            batch_time = AverageMeter()
            model_loss_meter = AverageMeter()
            recon_loss_meter = AverageMeter()
            kld_loss_meter = AverageMeter()
            end = time.time()

            for idx, [arch_code, illegal_arch_code] in enumerate(self.train_dataloader):
                self.adjust_learning_rate(self.opt_model, self.model_epoch, idx)
                arch_code = arch_code.to(self.device)
                if self.config.use_condition:
                    illegal_arch_code = illegal_arch_code.to(self.device)
                    x_cond = torch.zeros(arch_code.size(0) * 2).to(torch.int32).to(self.device)
                    x_cond[arch_code.size(0):] = 1
                    arch_code = torch.concatenate([arch_code, illegal_arch_code], dim=0)
                else:
                    x_cond = None

                arch_code = arch_code.unsqueeze(dim=1)
                model_loss, recon_loss, kld_loss = self.model(arch_code, condition=x_cond, vae_recon=self.vae_warmup)
                vae_loss = recon_loss + self.kld_weight * kld_loss
                loss = vae_loss if self.vae_warmup else (model_loss + vae_loss)

                self.opt_model.zero_grad()
                loss.backward()
                self.opt_model.step()

                model_loss_meter.update(model_loss.item(), arch_code.size(0))
                recon_loss_meter.update(recon_loss.item(), arch_code.size(0))
                kld_loss_meter.update(kld_loss.item(), arch_code.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % self.config.print_freq == 0 or idx + 1 == num_steps:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time.avg * (num_steps - idx)
                    model_name = 'VAE' if self.vae_warmup else 'Diffusion'
                    batch_log = f'{model_name} Train: [{self.model_epoch + 1}/{self.config.model_epochs}][{idx}/{num_steps}]\t' \
                            f'eta {timedelta(seconds=int(etas))}\t' \
                            f'model loss {model_loss_meter.val:.4f} ({model_loss_meter.avg:.4f})\t' \
                            f'recon loss {recon_loss_meter.val:.4f} ({recon_loss_meter.avg:.4f})\t' \
                            f'kld loss {kld_loss_meter.val:.4f} ({kld_loss_meter.avg:.4f})\t' \
                            f'mem {memory_used:.0f}MB'
                    self.logger.info(batch_log)

            if self.model_epoch + 1 >= int(0.2 * self.config.model_epochs) and self.vae_warmup:
                self.vae_warmup = False
                self.model_epoch = -1

            if (self.model_epoch + 1) % self.config.save_freq == 0:
                val_loss = self.validate_model()
                self.model_loss, is_best = min(self.model_loss, val_loss), val_loss < self.model_loss
                self.save_checkpoint(is_best)

            self.model_epoch += 1
        self.logger.info('Model Training Complete')

    def random_gen_arch_codes(self, sample_num):
        if self.config.use_condition:
            gen_x_cond = torch.zeros(sample_num).to(torch.int32).to(self.device)
        else:
            gen_x_cond = None
        latent_codes = torch.randn([sample_num, 4, 10]).to(self.device)
        arch_codes = self.model.p_sample(latent_codes, condition=gen_x_cond)
        arch_codes = arch_codes.reshape(sample_num, -1)

        return arch_codes, latent_codes

    def build_predictor_dataset(self, arch_num=1000):
        train_arch_codes, train_latent_codes = self.random_gen_arch_codes(arch_num)
        val_arch_codes, val_latent_codes = self.random_gen_arch_codes(int(arch_num * 0.25))
        train_dataset = PredictorDataset(train_arch_codes, train_latent_codes)
        val_dataset = PredictorDataset(val_arch_codes, val_latent_codes)

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_train * 4,
            shuffle=True,
            num_workers=self.config.num_worker
        )

        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_val * 4,
            shuffle=True,
            num_workers=self.config.num_worker
        )

    @torch.no_grad()
    def validate_predictor(self):
        self.model.eval()
        num_steps = len(self.val_dataloader)
        l_loss_meter = AverageMeter()
        e_loss_meter = AverageMeter()
        l_acc_meter = AverageMeter()
        e_acc_meter = AverageMeter()

        for idx, [layer_code, latent_code, runtime, energy] in enumerate(self.val_dataloader):
            latent_code = latent_code.to(self.device)
            condition_code = layer_code.to(self.device)
            runtime = runtime.unsqueeze(dim=1).to(self.device)
            energy = energy.unsqueeze(dim=1).to(self.device)

            l_pred, e_pred = self.predictor(latent_code, condition_code)
            l_loss = self.predictor_criterion(l_pred, runtime)
            e_loss = self.predictor_criterion(e_pred, energy)
            l_acc = biasd_accuracy(l_pred, runtime)
            e_acc = biasd_accuracy(e_pred, energy)
            l_loss_meter.update(l_loss, latent_code.size(0))
            e_loss_meter.update(e_loss, latent_code.size(0))
            l_acc_meter.update(l_acc, latent_code.size(0))
            e_acc_meter.update(e_acc, latent_code.size(0))
        
            if idx % self.config.print_freq == 0 or idx + 1 == num_steps:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                batch_log = f'Predictor Valid: [{self.predictor_epoch + 1}/{self.config.predictor_epochs}][{idx}/{num_steps}]\t' \
                        f'runtime loss {l_loss_meter.val:.4f} ({l_loss_meter.avg:.4f})\t' \
                        f'energy loss {e_loss_meter.val:.4f} ({e_loss_meter.avg:.4f})\t' \
                        f'runtime acc {l_acc_meter.val:.4f} ({l_acc_meter.avg:.4f})\t' \
                        f'energy acc {e_acc_meter.val:.4f} ({e_acc_meter.avg:.4f})\t' \
                        f'mem {memory_used:.0f}MB'
                self.logger.info(batch_log)
        
        return l_loss_meter.avg, e_loss_meter.avg

    def train_predictor(self):
        self.predictor.train()
        self.build_predictor_dataset()
        num_steps = len(self.train_dataloader)
        self._num_steps = num_steps
        self._num_epochs = self.config.predictor_epochs

        while self.predictor_epoch < self.config.predictor_epochs:
            l_loss_meter = AverageMeter()
            e_loss_meter = AverageMeter()
            batch_time = AverageMeter()
            end = time.time()

            for idx, [layer_code, latent_code, runtime, energy] in enumerate(self.train_dataloader):
                self.adjust_learning_rate(self.opt_predictor, self.predictor_epoch, idx)
                latent_code = latent_code.to(self.device)
                condition_code = layer_code.to(self.device)
                runtime = runtime.unsqueeze(dim=1).to(self.device)
                energy = energy.unsqueeze(dim=1).to(self.device)

                l_pred, e_pred = self.predictor(latent_code, condition_code)
                l_loss = self.predictor_criterion(l_pred, runtime)
                e_loss = self.predictor_criterion(e_pred, energy)

                self.opt_predictor.zero_grad()
                l_loss.backward()
                e_loss.backward()
                self.opt_predictor.step()

                l_loss_meter.update(l_loss, latent_code.size(0))
                e_loss_meter.update(e_loss, latent_code.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % self.config.print_freq == 0 or idx + 1 == num_steps:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time.avg * (num_steps - idx)
                    batch_log = f'Predictor Train: [{self.predictor_epoch + 1}/{self.config.predictor_epochs}][{idx}/{num_steps}]\t' \
                            f'eta {timedelta(seconds=int(etas))}\t' \
                            f'runtime loss {l_loss_meter.val:.4f} ({l_loss_meter.avg:.4f})\t' \
                            f'energy loss {e_loss_meter.val:.4f} ({e_loss_meter.avg:.4f})\t' \
                            f'mem {memory_used:.0f}MB'
                    self.logger.info(batch_log)
            
            if (self.predictor_epoch + 1) % self.config.save_freq == 0:
                val_l_loss, val_e_loss = self.validate_predictor()
                val_loss = (val_l_loss + val_e_loss) / 2
                self.predictor_loss, is_best = min(self.predictor_loss, val_loss), val_loss <= self.predictor_loss
                self.save_checkpoint(is_best)
            self.predictor_epoch += 1
        self.logger.info('Predictor Training Complete')
