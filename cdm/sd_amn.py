import re
import torch
import timm

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from utils.util import cal_anomaly_map, log_local, create_logger
from utils.eval_helper import dump, log_metrics, merge_together, performances, save_metrics
from cdm.param import no_trained_para, control_trained_para, contains_any, sub_
from cdm.mha import MultiheadAttention

import os

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cdm.vit import *


class SD_AMN(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, layers, distance, log_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.automatic_optimization = False

        self.control_model = instantiate_from_config(control_stage_config)

        self.control_key = control_key

        self.log_name = log_name

        self.pretrained_resnet50 = timm.create_model("resnet50", pretrained=True, features_only=True)  # for inference

        self.layers_ = layers  # for inference

        self.distance = distance  # for inference

        self.criterion_mse = nn.MSELoss()

        self.result = {'clsname': [], 'filename': [], 'pred': [], 'mask': []}

        self.max_check = 0.0
        self.task_id = 0

        self.set_train_param()

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())

        for name, p in self.model.diffusion_model.named_parameters():
            if contains_any(name, no_trained_para):
                # print(name)
                params.append(p)

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def set_log_name(self, log_name):
        self.log_name = log_name

        if hasattr(self, 'logger_val'):
            for handler in self.logger_val.handlers[:]:
                self.logger_val.removeHandler(handler)
                handler.close()

        os.makedirs(os.path.dirname(f"logs/{self.log_name}"), exist_ok=True)
        os.makedirs(os.path.dirname(f"project/{self.log_name}"), exist_ok=True)
        self.logger_val = create_logger("global_logger", f"logs/{self.log_name}/")

    def set_train_param(self):
        self.unet_train_param_name = []
        self.control_train_param_name = control_trained_para

        for name, p in self.model.diffusion_model.named_parameters():
            if contains_any(name, no_trained_para) and not sub_(name) in self.unet_train_param_name:
                self.unet_train_param_name.append(sub_(name))

    def training_step(self, batch, batch_idx):

        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)

        if not self.task_id == 0:

            for name, param in self.control_model.named_parameters():
                if sub_(name) in self.project.keys() and re.search('.weight', name):
                    param.grad.data = param.grad.data - torch.mm(param.grad.data.view(param.shape[0], -1),
                                                                 torch.mm(self.project[sub_(name)],
                                                                          self.project[sub_(name)].t())).view(
                        param.shape)
                else:
                    param.grad.data.fill_(0.0)

            for name, param in self.model.diffusion_model.named_parameters():
                if sub_(name) in self.project.keys() and re.search('.weight', name):
                    param.grad.data = (param.grad.data - torch.mm(param.grad.data.view(param.shape[0], -1),
                                                                  torch.mm(self.project[sub_(name)],
                                                                           self.project[sub_(name)].t())).view(
                        param.shape))
                else:
                    param.grad.data.fill_(0.0)

        opt.step()

        # return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        output = self.log_images_test(batch)

        input_img = output['reconstruction']

        self.pretrained_resnet50.eval()
        input_features = self.pretrained_resnet50(input_img)

        log_local(output, batch["filename"], f'log_image/{self.log_name}')
        output_img = output['samples']
        output_features = self.pretrained_resnet50(output_img)

        input_features = [input_features[i] for i in self.layers_]
        output_features = [output_features[i] for i in self.layers_]

        anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a',
                                         dis=self.distance)

        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=5)
        anomaly_map = torch.from_numpy(anomaly_map)
        anomaly_map_prediction = anomaly_map.unsqueeze(dim=1)

        self.result['filename'] += batch["filename"]  # list:12
        self.result['pred'] += list(anomaly_map_prediction.cpu().numpy())  # B x 1 x H x W
        self.result['mask'] += list(batch["mask"].cpu().numpy())  # B x 1 x H x W
        self.result['clsname'] += batch["clsname"]  # list:12

    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs):

        # evl_metrics = {'auc': [{'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'apsp'}]}
        evl_metrics = {
            'auc': [
                {'name': 'max', 'kwargs': {'topk_ratio': 0.01, 'smooth_kernel': 4}},
                {'name': 'pixel'}
            ]
        }
        self.print("Gathering final results ...")
        fileinfos, preds, masks = merge_together(self.result)

        self.result['filename'].clear()
        self.result['pred'].clear()
        self.result['mask'].clear()
        self.result['clsname'].clear()

        ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
        log_metrics(ret_metrics, evl_metrics)
        auroc_px = ret_metrics['mean_pixel_auc']
        auroc_sp = ret_metrics['mean_max_auc']
        val_acc = auroc_px + auroc_sp
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        if val_acc > self.max_check:
            self.max_check = val_acc
            save_metrics(ret_metrics, evl_metrics, f'logs/{self.log_name}-best.csv')

    # def get_activation(self, name):
    #     def hook(model, input, output):
    #         if (isinstance(model, nn.Linear)
    #                 or isinstance(model, nn.modules.linear.NonDynamicallyQuantizableLinear)
    #                 or isinstance(model, MultiheadAttention)):

    #             input_channel = input[0].shape[-1]

    #             mat = input[0].reshape(-1, input_channel).t().cpu()

    #             if name in self.act.keys():
    #                 self.act[name] = torch.cat([self.act[name], mat], dim=1)
    #             else:
    #                 self.act[name] = mat

    #         elif isinstance(model, nn.Conv2d):
    #             batch_size, input_channel, input_map_size, _ = input[0].shape
    #             padding = model.padding[0]
    #             kernel_size = model.kernel_size[0]
    #             stride = model.stride[0]

    #             mat = F.unfold(input[0], kernel_size=kernel_size, stride=stride, padding=padding).transpose(0, 1).reshape(kernel_size*kernel_size*input_channel, -1).detach().cpu()

    #             if name in self.act.keys():
    #                 self.act[name] = torch.cat([self.act[name], mat], dim=1)
    #             else:
    #                 self.act[name] = mat

    #     return hook

    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
    #     if batch_idx % 10 == 0:
    #         _ = self.log_images_test(batch)

    # @torch.no_grad()
    # def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     if batch_idx % 10 == 0:
    #         for name, act in self.act.items():
    #             U, S, Vh = torch.linalg.svd(act.cuda(), full_matrices=False)

    #             sval_total = (S**2).sum()
    #             sval_ratio = (S**2) / sval_total
    #             r = max(torch.sum(torch.cumsum(sval_ratio, dim=0) < 0.999), 1)
    #             self.act[name] = U[:, :r].cpu()

    # @torch.no_grad()
    # def on_test_end(self):
    #     for name, act in self.act.items():
    #         if not name in self.project.keys():
    #             self.project[name] = act.cuda()
    #         else:
    #             U1, S1, Vh1 = torch.linalg.svd(act.cuda(), full_matrices=False)

    #             sval_total = (S1**2).sum()

    #             act_hat = act.cuda() - self.project[name] @ self.project[name].t() @ act.cuda()

    #             U, S, Vh = torch.linalg.svd(act_hat)

    #             sval_hat = (S**2).sum()
    #             sval_ratio = (S**2) / sval_total

    #             accumulated_sval = (sval_total - sval_hat) / sval_total

    #             r = 0
    #             for ii in range(sval_ratio.shape[0]):
    #                 # if accumulated_sval < 0.99:
    #                 if accumulated_sval < 0.5:
    #                     accumulated_sval += sval_ratio[ii]
    #                     r += 1
    #                 else:
    #                     break

    #             self.project[name] = torch.cat([self.project[name], act_hat[:, :r]], dim=1)

    #     torch.save(self.project, f"project/{self.log_name}.pt")

    #     for value in self.hook_handle.values():
    #         value.remove()

    #     self.task_id += 1
    #     self.max_check = 0.0

    # @torch.no_grad()
    # def on_test_start(self):

    #     self.hook_handle = {}
    #     del self.act
    #     self.act = {}
    #     for name, module in self.model.diffusion_model.named_modules():
    #         if name in self.unet_train_param_name:
    #             self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))

    #     for name, module in self.control_model.named_modules():
    #         if name in self.control_train_param_name:
    #             self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c, init_img = super().get_input(batch, k, *args, **kwargs)
        control = batch[self.control_key]
        control = control.to(self.device)
        control = control.to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], c_concat=[control], init=[init_img])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            assert 'c_oncat is None'
        else:
            control = self.control_model(hint=torch.cat(cond['c_concat'], 1))
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)

        return eps

    def apply_model_train(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            assert 'c_oncat is None'
        else:
            control = self.control_model(hint=torch.cat(cond['c_concat'], 1))
            hint_f = self.control_model(hint=torch.cat(cond['init'], 1), re=False)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)

            re_loss = self.criterion_mse(control, hint_f)

        return eps, re_loss

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images_test(self, batch, sample=False, ddim_steps=10, ddim_eta=0.0, plot_denoise_rows=False,
                        unconditional_guidance_scale=9.0):
        use_ddim = ddim_steps is not None

        N = batch['jpg'].shape[0]

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat = c["c_concat"][0][:N]
        c = c["c_crossattn"][0][:N]
        log["reconstruction"] = self.decode_first_stage(z)

        t = torch.randint(999, 1000, (z.shape[0],), device=self.device).long()
        noise = torch.randn_like(z)
        x_noisy = self.q_sample(x_start=z, t=t, noise=noise)

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, inter = self.sample_log_test(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                      batch_size=N, ddim=use_ddim,
                                                      ddim_steps=ddim_steps, eta=ddim_eta,
                                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                                      unconditional_conditioning=uc_full,
                                                      x_T=x_noisy, timesteps=t
                                                      )
            log["samples"] = self.decode_first_stage(samples_cfg)
            # for i in range(1, 101):
            #     log[f'samples{i}'] = self.decode_first_stage(inter['x_inter'][i])

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def sample_log_test(self, cond, batch_size, ddim, ddim_steps, x_T, timesteps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, x_T, timesteps, verbose=False,
                                                     **kwargs)
        return samples, intermediates

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()