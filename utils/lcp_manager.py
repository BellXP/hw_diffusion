from utils.run_manager import *
import pickle
import numpy as np


def satisfy_arch_constraint(arch_code):
    arch_code = arch_code.reshape(-1, 5, 4)
    for i in range(arch_code.shape[1] - 1):
        if arch_code[:, i, :].sum() < arch_code[:, i+1, :].sum():
            return False
    return True


class LCPManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.run_manager = RunManager(config, logger, True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.run_manager.model
        self.predictor = self.run_manager.predictor
        self.model.eval()
        self.predictor.eval()
        self.predictor_criterion = get_criterion(config.pred_loss)
    
        ckpt = pickle.load(open('/nvme/xupeng/workplace/dataset/linear_try/linear_try_data.pkl', 'rb'))
        self.layer_defs = ckpt['layer_codes']
        self.energy_line_kernel = ckpt['energy_line_kernel']
        self.runtime_line_kernel = ckpt['runtime_line_kernel']
        self.arch_scale = [10, 20, 30, 40] * 5

    def get_latent_code_grad(self, target, latent_code, condition_code):
        l_latent_code_grad, e_latent_code_grad = 0, 0
        self.predictor.zero_grad()
        latent_code.requires_grad = True
        l_pred, e_pred = self.predictor(latent_code, condition_code)
        if 'latency' in target:
            l_target = l_pred.clone().detach() - self.config.prop_step
            l_loss = self.predictor_criterion(l_pred, l_target)
            l_latent_code_grad = torch.autograd.grad(l_loss, latent_code, retain_graph=True)[0]
        if 'energy' in target:
            e_target = e_pred.clone().detach() - self.config.prop_step
            e_loss = self.predictor_criterion(e_pred, e_target)
            e_latent_code_grad = torch.autograd.grad(e_loss, latent_code, retain_graph=True)[0]
        latent_code_grad = l_latent_code_grad + e_latent_code_grad
        if 'latency' in target and 'energy' in target:
            latent_code_grad = 0.5 * latent_code_grad

        return latent_code_grad
    
    def get_real_perf(self, latent_code, condition_code):
        assert latent_code.shape[0] == condition_code.shape[0]
        arch_codes = np.array(self.decode_latent_code(latent_code))
        condition_code = condition_code.view(condition_code.shape[0], -1)[:, :7].detach().cpu().numpy()
        data_input = np.concatenate([arch_codes, condition_code], axis=1)
        data_energy = (data_input * self.energy_line_kernel).sum(axis=1)
        data_runtime = (data_input * self.runtime_line_kernel).sum(axis=1)

        feasible_check = [satisfy_arch_constraint(arch_code) for arch_code in arch_codes]

        return data_energy, data_runtime, feasible_check

    def decode_latent_code(self, latent_code):
        if self.config.use_condition:
            x_cond = torch.zeros(latent_code.size(0)).to(torch.int32).to(self.device)
        else:
            x_cond = None
        arch_code = self.model.p_sample(latent_code, condition=x_cond).squeeze(dim=1)
        
        return arch_code.long().tolist()

    def layer_lcp(self, layer_num=3, target='latency#energy'):
        layer_codes = self.layer_defs[:layer_num]
        condition_codes = torch.tensor(layer_codes).view(-1, 7).float().to(self.device)
        latent_codes = torch.randn(condition_codes.size(0), 4, 10).to(self.device)
        ini_latent_codes = latent_codes.clone().detach()

        for _ in range(self.config.prop_epoch):
            latent_code_grads = self.get_latent_code_grad(target, latent_codes, condition_codes)
            latent_codes = latent_codes - latent_code_grads * self.config.prop_lr
            latent_codes = latent_codes.clamp(-1, 1).clone().detach()

        ini_energy, ini_runtime, ini_feasible_check = self.get_real_perf(ini_latent_codes, condition_codes)
        print(f'Ini perf: energy {ini_energy}, runtime {ini_runtime}, feasible {ini_feasible_check}')
        lcp_energy, lcp_runtime, lcp_feasible_check = self.get_real_perf(latent_codes, condition_codes)
        print(f'LCP perf: energy {lcp_energy}, runtime {lcp_runtime}, feasible {lcp_feasible_check}')

        arch_codes = self.decode_latent_code(latent_codes)
        self.logger.info('Layer Latent Code Propagation Complete')

        return arch_codes, layer_codes
