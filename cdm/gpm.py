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

from cdm.sd_amn import SD_AMN
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cdm.vit import *
from cdm.lora_expert import AdditiveLoRAAdapter


class CDAD(SD_AMN):
    """
    The implementation of GPM and iSVD
    """

    def __init__(self, lora_rank=4, lora_alpha=1.0, orth_lambda=1e-4, novelty_threshold=0.2,
                 init_experts=1, max_experts_per_layer=8, max_new_experts_per_task=2, novelty_step=0.1,
                 *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.project = {}
            self.act = {}
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.orth_lambda = orth_lambda
            self.novelty_threshold = novelty_threshold
            self.init_experts = init_experts
            self.max_experts_per_layer = max_experts_per_layer
            self.max_new_experts_per_task = max_new_experts_per_task
            self.novelty_step = novelty_step
            self.layer_adapters = {}
            self._attach_lora_adapters()
            self._append_new_expert(trainable=True, num_new_experts=self.init_experts, freeze_previous=False)

    def _attach_lora_adapters(self):
        target_names = set(self.unet_train_param_name + self.control_train_param_name)
        for name, module in self.model.diffusion_model.named_modules():
            if name in target_names and isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.split('.')[-1]
                parent = self.model.diffusion_model.get_submodule(
                    parent_name) if parent_name else self.model.diffusion_model
                adapter = AdditiveLoRAAdapter(module, rank=self.lora_rank, alpha=self.lora_alpha)
                setattr(parent, attr_name, adapter)
                self.layer_adapters[f"diffusion::{name}"] = adapter

        for name, module in self.control_model.named_modules():
            if name in target_names and isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.split('.')[-1]
                parent = self.control_model.get_submodule(parent_name) if parent_name else self.control_model
                adapter = AdditiveLoRAAdapter(module, rank=self.lora_rank, alpha=self.lora_alpha)
                setattr(parent, attr_name, adapter)
                self.layer_adapters[f"control::{name}"] = adapter

    def _append_new_expert(self, trainable=True, num_new_experts=1, freeze_previous=True):
        if num_new_experts <= 0:
            return
        for adapter in self.layer_adapters.values():
            if freeze_previous and adapter.num_experts > 0:
                adapter.freeze_expert(adapter.num_experts - 1)
            for _ in range(num_new_experts):
                adapter.add_expert(trainable=trainable)

    def _orthogonal_regularization(self):
        if not self.layer_adapters:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for adapter in self.layer_adapters.values():
            loss = loss + adapter.orthogonality_loss()
        return loss

    def _novelty_energy(self):
        energies = []
        for adapter in self.layer_adapters.values():
            if adapter.num_experts == 0:
                continue
            energies.append(adapter.novelty_energy(adapter.num_experts - 1))
        if len(energies) == 0:
            return 0.0
        return float(sum(energies) / len(energies))

    def configure_optimizers(self):
        lr = self.learning_rate
        trainable = []
        for adapter in self.layer_adapters.values():
            for i in range(adapter.num_experts):
                a, b = adapter._get_ab(i)
                g = adapter.expert_gates[i]
                if a.requires_grad:
                    trainable.extend([a, b, g])

        for p in self.control_model.parameters():
            p.requires_grad = False

        for p in self.model.diffusion_model.parameters():
            p.requires_grad = False

        for p in trainable:
            p.requires_grad = True

        if len(trainable) == 0:
            raise RuntimeError(
                "No trainable LoRA parameters were found. "
                "Please check whether LoRA adapters were attached to any target layers."
            )

        opt = torch.optim.AdamW(trainable, lr=lr)
        return opt

    @torch.no_grad()
    def fuse_lora_experts(self):
        for adapter in self.layer_adapters.values():
            adapter.fuse_into_weight()

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
        orth_loss = self._orthogonal_regularization()
        total_loss = loss + self.orth_lambda * orth_loss

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("orth_loss", orth_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

    def get_activation(self, name):
        def hook(model, input, output):
            if (isinstance(model, nn.Linear)
                    or isinstance(model, nn.modules.linear.NonDynamicallyQuantizableLinear)
                    or isinstance(model, MultiheadAttention)):

                input_channel = input[0].shape[-1]

                mat = input[0].reshape(-1, input_channel).t().cpu()

                if name in self.act.keys():
                    self.act[name] = torch.cat([self.act[name], mat], dim=1)
                else:
                    self.act[name] = mat

            elif isinstance(model, nn.Conv2d):
                batch_size, input_channel, input_map_size, _ = input[0].shape
                padding = model.padding[0]
                kernel_size = model.kernel_size[0]
                stride = model.stride[0]

                mat = F.unfold(input[0], kernel_size=kernel_size, stride=stride, padding=padding).transpose(0,
                                                                                                            1).reshape(
                    kernel_size * kernel_size * input_channel, -1).detach().cpu()

                if name in self.act.keys():
                    self.act[name] = torch.cat([self.act[name], mat], dim=1)
                else:
                    self.act[name] = mat

        return hook

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            _ = self.log_images_test(batch)

    @torch.no_grad()
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            for name, act in self.act.items():
                U, S, Vh = torch.linalg.svd(act.cuda(), full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = max(torch.sum(torch.cumsum(sval_ratio, dim=0) < 0.999), 1)
                self.act[name] = U[:, :r].cpu()

    @torch.no_grad()
    def on_test_end(self):
        novelty = self._novelty_energy()
        grow = novelty > self.novelty_threshold
        num_new_experts = 0
        if grow:
            step = max(self.novelty_step, 1e-8)
            num_new_experts = 1 + int((novelty - self.novelty_threshold) / step)
            num_new_experts = min(num_new_experts, self.max_new_experts_per_task)

            if self.max_experts_per_layer is not None:
                current = 0
                if len(self.layer_adapters) > 0:
                    current = min(adapter.num_experts for adapter in self.layer_adapters.values())
                room = max(self.max_experts_per_layer - current, 0)
                num_new_experts = min(num_new_experts, room)

        if num_new_experts > 0:
            self._append_new_expert(trainable=True, num_new_experts=num_new_experts, freeze_previous=True)

        for value in self.hook_handle.values():
            value.remove()

        self.logger_val.info(
            f"LoRA novelty={novelty:.4f}, threshold={self.novelty_threshold:.4f}, "
            f"grow={grow}, new_experts={num_new_experts}"
        )
        self.task_id += 1
        self.max_check = 0.0

    @torch.no_grad()
    def on_test_start(self):

        self.hook_handle = {}
        del self.act
        self.act = {}
        for name, module in self.model.diffusion_model.named_modules():
            if name in self.unet_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))

        for name, module in self.control_model.named_modules():
            if name in self.control_train_param_name:
                self.hook_handle[name] = module.register_forward_hook(self.get_activation(name))
