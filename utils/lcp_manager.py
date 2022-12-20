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
        self.run_manager = RunManager(config, logger)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.run_manager.model
        self.latency_predictor = self.run_manager.latency_predictor
        self.energy_predictor = self.run_manager.energy_predictor
        self.latency_predictor.eval()
        self.energy_predictor.eval()
        self.predictor_criterion = get_criterion(config.pred_loss)
    
        # self.model_defs = pickle.load(open(config.model_defs_path, 'rb'))
        # self.layer_defs = np.load(config.layer_defs_path)
        ckpt = pickle.load(open('/nvme/xupeng/workplace/dataset/linear_try/linear_try_data.pkl', 'rb'))
        self.layer_defs = ckpt['layer_codes']
        self.energy_line_kernel = ckpt['energy_line_kernel']
        self.runtime_line_kernel = ckpt['runtime_line_kernel']
        self.arch_scale = [10, 20, 30, 40] * 5

    def get_latent_code_grad(self, target, latent_code, condition_code):
        l_latent_code_grad, e_latent_code_grad = 0, 0
        if 'latency' in target:
            self.latency_predictor.zero_grad()
            latent_code.requires_grad = True
            l_pred = self.latency_predictor(latent_code, condition_code)
            l_target = l_pred.clone().detach() - self.config.prop_step
            l_loss = self.predictor_criterion(l_pred, l_target)
            l_latent_code_grad = torch.autograd.grad(l_loss, latent_code, retain_graph=True)[0]
        if 'energy' in target:
            self.energy_predictor.zero_grad()
            latent_code.requires_grad = True
            e_pred = self.energy_predictor(latent_code, condition_code)
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

        if self.config.use_diffusion:
            latent_code = latent_code.unsqueeze(dim=1)
            arch_code = self.model.p_sample_loop(latent_code, condition=x_cond)
        else:
            arch_code = self.model.decode(latent_code, x_cond)
        
        return arch_code.long().tolist()

    def layer_lcp(self, model_names, target='latency#energy'):
        # model_layers = []
        # for model_name in model_names:
        #     model_layers.append(self.model_defs[model_name])
        # model_layers = sum(model_layers, [])
        # model_layers = list(set(model_layers))
        # layer_codes = [self.layer_defs[x] for x in model_layers]
        layer_codes = self.layer_defs

        condition_codes = torch.tensor(layer_codes).view(-1, self.config.condition_dim).float().to(self.device)
        if self.config.log_layercode:
            condition_codes = torch.log(condition_codes)
        latent_codes = torch.randn(condition_codes.size(0), self.config.latent_dim).to(self.device)
        ini_latent_codes = latent_codes.clone().detach()

        for _ in range(self.config.prop_epoch):
            latent_code_grads = self.get_latent_code_grad(target, latent_codes, condition_codes)
            latent_codes = latent_codes - latent_code_grads * self.config.prop_lr
            latent_codes = latent_codes.clamp(-1, 1).clone().detach()

        ini_energy, ini_runtime, ini_feasible_check = self.get_real_perf(ini_latent_codes, condition_codes)
        print(f'Ini perf: energy {ini_energy}, feasible {ini_feasible_check}')
        lcp_energy, lcp_runtime, lcp_feasible_check = self.get_real_perf(latent_codes, condition_codes)
        print(f'LCP perf: energy {lcp_energy}, feasible {lcp_feasible_check}')

        arch_codes = self.decode_latent_code(latent_codes)
        self.logger.info('Layer Latent Code Propagation Complete')

        return arch_codes, layer_codes

    def model_lcp(self, model_names, target='latency#energy'):
        model_lcp_results = []
        for model_name in model_names:
            model_layers = self.model_defs[model_name]
            layer_codes = [self.layer_defs[x] for x in model_layers]
            condition_codes = torch.tensor(layer_codes).view(-1, self.config.condition_dim).float().to(self.device)
            if self.config.log_layercode:
                condition_codes = torch.log(condition_codes)
            latent_code = torch.randn(1, self.config.latent_dim).to(self.device)

            for _ in range(self.config.prop_epoch):
                multi_latent_codes = torch.repeat_interleave(latent_code, repeats=condition_codes.size(0), dim=0)
                multi_latent_code_grads = self.get_latent_code_grad(target, multi_latent_codes, condition_codes)
                latent_code_grad = multi_latent_code_grads.mean(dim=0).view(1, self.config.latent_dim)
                latent_code = latent_code - latent_code_grad * self.config.prop_lr
                latent_code = latent_code.clamp(-1, 1)

            arch_code = self.decode_latent_code(latent_code)
            model_lcp_results.append([arch_code, layer_codes])
            self.logger.info(f'{model_name} Model Latent Code Propagation Complete')

        return model_lcp_results