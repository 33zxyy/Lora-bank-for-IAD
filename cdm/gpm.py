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
                 orth_constraint_mode="soft", adaptive_init_on_task0=False, router_topk=2,
                 warmup_novelty_batches=4, conv_sample_pool_stride=2,
                 conv_sample_max_patches=512, warmup_max_cols_per_layer=2048,
                 layer_growth_topk=4, highres_threshold_scale=0.8, middle_attn_threshold_scale=1.2,
                 warmup_stat_layer_keywords=("input_blocks", "middle_block", "output_blocks"),
                 warmup_stat_max_layers=24,
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
            self.adaptive_init_on_task0 = adaptive_init_on_task0
            self.router_topk = router_topk
            self.warmup_novelty_batches = warmup_novelty_batches
            self.conv_sample_pool_stride = max(int(conv_sample_pool_stride), 1)
            self.conv_sample_max_patches = max(int(conv_sample_max_patches), 1)
            self.warmup_max_cols_per_layer = max(int(warmup_max_cols_per_layer), 1)
            self.layer_growth_topk = max(int(layer_growth_topk), 0)
            self.highres_threshold_scale = float(highres_threshold_scale)
            self.middle_attn_threshold_scale = float(middle_attn_threshold_scale)
            self.warmup_stat_layer_keywords = tuple(warmup_stat_layer_keywords or ())
            self.warmup_stat_max_layers = max(int(warmup_stat_max_layers), 1)
            self._warmup_batch_count = 0
            self._warmup_done = False
            self._warmup_hook_handle = {}
            if orth_constraint_mode not in ("soft", "hard"):
                raise ValueError(f"orth_constraint_mode must be 'soft' or 'hard', got {orth_constraint_mode}")
            self.orth_constraint_mode = orth_constraint_mode
            self.layer_adapters = {}
            self._attach_lora_adapters()
            self._assert_control_branch_without_lora()
            init_num_experts = self.init_experts
            if self.adaptive_init_on_task0:
                # Before task0 we do not have previous task statistics yet.
                # Use maximal novelty proxy (1.0) with the same growth rule.
                init_num_experts = self._compute_new_experts_from_novelty(1.0, current_experts=0)
                init_num_experts = max(init_num_experts, 1)
            self._append_new_expert(trainable=True, num_new_experts=init_num_experts, freeze_previous=False)

    def _attach_lora_adapters(self):
        # Keep LoRA on diffusion UNet only; AMN/control branch runs without LoRA.
        target_names = set(self.unet_train_param_name)
        for name, module in self.model.diffusion_model.named_modules():
            if name in target_names and isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.split('.')[-1]
                parent = self.model.diffusion_model.get_submodule(
                    parent_name) if parent_name else self.model.diffusion_model
                adapter = AdditiveLoRAAdapter(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    max_experts=max(self.max_experts_per_layer or 0, 32)
                )
                adapter.orth_mode = self.orth_constraint_mode
                adapter.router_topk = self.router_topk
                setattr(parent, attr_name, adapter)
                self.layer_adapters[f"diffusion::{name}"] = adapter

    def _append_new_expert(self, trainable=True, num_new_experts=1, freeze_previous=True):
        if num_new_experts <= 0:
            return
        for key, adapter in self.layer_adapters.items():
            if freeze_previous and adapter.num_experts > 0:
                for i in range(adapter.num_experts):
                    adapter.freeze_expert(i)
            for _ in range(num_new_experts):
                adapter.add_expert(trainable=trainable)
                adapter.to(adapter.module.weight.device)
                layer_name = key.split("::", 1)[-1]
                self._init_new_expert_from_residual(adapter, layer_name, adapter.num_experts - 1)

    def _init_new_expert_from_residual(self, adapter, layer_name, expert_idx):
        device = adapter.module.weight.device
        dtype = adapter.module.weight.dtype
        with torch.no_grad():
            a, b = adapter._get_ab(expert_idx)
            # Random init + projection to the orthogonal complement of previous experts.
            a_rand = torch.empty_like(a)
            nn.init.kaiming_uniform_(a_rand, a=5 ** 0.5)

            u_prev = adapter.orth_basis(upto=expert_idx)
            if u_prev is not None and u_prev.numel() > 0:
                u_prev = u_prev.to(device=device, dtype=dtype)
                a_rand = a_rand - (a_rand @ u_prev.t()) @ u_prev

            a.data.copy_(a_rand)
            b.data.zero_()
            adapter.expert_gates[expert_idx].data.fill_(-2.0)

    def _assert_control_branch_without_lora(self):
        for module in self.control_model.modules():
            if isinstance(module, AdditiveLoRAAdapter):
                raise RuntimeError("AMN/control branch contains LoRA adapters, but it should remain LoRA-free.")

    def _start_task_warmup(self):
        self._warmup_batch_count = 0
        self._warmup_done = (self.warmup_novelty_batches is None or self.warmup_novelty_batches <= 0)
        self._warmup_hook_handle = {}
        if self._warmup_done:
            return
        self.act = {}
        stat_layers = self._select_warmup_stat_layers()
        for name, module in self.model.diffusion_model.named_modules():
            if name in stat_layers:
                self._warmup_hook_handle[name] = module.register_forward_hook(self.get_activation(name))

    def _finish_task_warmup_and_grow(self):
        self._finalize_warmup_activations()
        layer_novelty = self._layer_novelty_energy()
        ranked = []
        for key, novelty in layer_novelty.items():
            threshold = self._layer_novelty_threshold(key)
            if novelty > threshold:
                ranked.append((key, novelty, threshold))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if self.layer_growth_topk > 0:
            ranked = ranked[:self.layer_growth_topk]

        growth_plan = {}
        for key, novelty, threshold in ranked:
            current = self.layer_adapters[key].num_experts
            num_new_experts = self._compute_new_experts_from_novelty(
                novelty, current_experts=current, threshold=threshold
            )
            if num_new_experts > 0:
                growth_plan[key] = num_new_experts

        if len(growth_plan) > 0:
            self._append_new_expert_by_plan(growth_plan, trainable=True, freeze_previous=True)
            self._register_new_trainable_params_to_optimizer()
        for value in self._warmup_hook_handle.values():
            value.remove()
        self._warmup_hook_handle = {}
        self.act = {}
        self._warmup_done = True
        avg_novelty = 0.0 if len(layer_novelty) == 0 else float(sum(layer_novelty.values()) / len(layer_novelty))
        self.logger_val.info(
            f"[WarmupGrow] avg_novelty={avg_novelty:.4f}, grown_layers={len(growth_plan)}, "
            f"growth_plan={growth_plan}, warmup_batches={self._warmup_batch_count}"
        )

    def _select_warmup_stat_layers(self):
        adapter_layers = [key.split("::", 1)[-1] for key in self.layer_adapters.keys()]
        if len(adapter_layers) == 0:
            return set()
        selected = []
        for name in adapter_layers:
            if len(self.warmup_stat_layer_keywords) == 0:
                selected.append(name)
            elif any(k in name for k in self.warmup_stat_layer_keywords):
                selected.append(name)
        if len(selected) == 0:
            selected = adapter_layers
        if len(selected) > self.warmup_stat_max_layers:
            stride = max(len(selected) // self.warmup_stat_max_layers, 1)
            selected = selected[::stride][:self.warmup_stat_max_layers]
        return set(selected)

    def _finalize_warmup_activations(self):
        for name, chunks in list(self.act.items()):
            if isinstance(chunks, list):
                if len(chunks) == 0:
                    self.act[name] = torch.empty(0, 0, device=self.device)
                    continue
                mat = torch.cat(chunks, dim=1)
                if mat.shape[1] > self.warmup_max_cols_per_layer:
                    idx = torch.randperm(mat.shape[1], device=mat.device)[:self.warmup_max_cols_per_layer]
                    mat = mat[:, idx]
                self.act[name] = mat

    def _append_new_expert_by_plan(self, growth_plan, trainable=True, freeze_previous=True):
        for key, num_new_experts in growth_plan.items():
            if num_new_experts <= 0 or key not in self.layer_adapters:
                continue
            adapter = self.layer_adapters[key]
            if freeze_previous and adapter.num_experts > 0:
                for i in range(adapter.num_experts):
                    adapter.freeze_expert(i)
            for _ in range(num_new_experts):
                adapter.add_expert(trainable=trainable)
                adapter.to(adapter.module.weight.device)
                layer_name = key.split("::", 1)[-1]
                self._init_new_expert_from_residual(adapter, layer_name, adapter.num_experts - 1)

    def _register_new_trainable_params_to_optimizer(self):
        if self.trainer is None:
            return
        opt = self.optimizers()
        if opt is None:
            return
        existing = {id(p) for group in opt.param_groups for p in group['params']}
        new_params = []
        for adapter in self.layer_adapters.values():
            for i in range(adapter.num_experts):
                a, b = adapter._get_ab(i)
                g = adapter.expert_gates[i]
                for p in (a, b, g):
                    if p.requires_grad and id(p) not in existing:
                        new_params.append(p)
            for p in adapter.router.parameters():
                if p.requires_grad and id(p) not in existing:
                    new_params.append(p)
        if len(new_params) > 0:
            opt.add_param_group({"params": new_params})

    def _orthogonal_regularization(self):
        if not self.layer_adapters:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for adapter in self.layer_adapters.values():
            loss = loss + adapter.orthogonality_loss()
        return loss

    def _layer_novelty_threshold(self, layer_key):
        layer_name = layer_key.split("::", 1)[-1]
        threshold = self.novelty_threshold
        if "input_blocks" in layer_name or "output_blocks" in layer_name:
            threshold = threshold * self.highres_threshold_scale
        if "middle_block" in layer_name or "attn" in layer_name:
            threshold = threshold * self.middle_attn_threshold_scale
        return float(max(threshold, 1e-8))

    def _layer_novelty_energy(self):
        energies = {}
        for key, adapter in self.layer_adapters.items():
            if adapter.num_experts == 0:
                continue
            layer_name = key.split("::", 1)[-1]
            if layer_name not in self.act:
                continue
            feat = self.act[layer_name].to(device=self.device)
            if feat.numel() == 0:
                continue
            u_old = adapter.orth_basis(upto=adapter.num_experts)
            if u_old is None or u_old.numel() == 0:
                energies[key] = 1.0
                continue
            u_old = u_old.to(device=self.device)
            proj = u_old.t() @ (u_old @ feat)
            residual = feat - proj
            den = torch.norm(feat, p='fro') ** 2 + 1e-8
            num = torch.norm(residual, p='fro') ** 2
            energies[key] = float((num / den).item())
        return energies

    def _novelty_energy(self):
        energies = self._layer_novelty_energy()
        if len(energies) == 0:
            return 0.0
        return float(sum(energies.values()) / len(energies))

    def _compute_new_experts_from_novelty(self, novelty, current_experts, threshold=None):
        threshold = self.novelty_threshold if threshold is None else float(threshold)
        if novelty <= threshold:
            return 0
        step = max(self.novelty_step, 1e-8)
        num_new_experts = 1 + int((novelty - threshold) / step)
        num_new_experts = min(num_new_experts, self.max_new_experts_per_task)

        if self.max_experts_per_layer is not None:
            room = max(self.max_experts_per_layer - current_experts, 0)
            num_new_experts = min(num_new_experts, room)
        return max(num_new_experts, 0)

    def configure_optimizers(self):
        lr = self.learning_rate
        lora_trainable = []
        for adapter in self.layer_adapters.values():
            for p in adapter.router.parameters():
                if p.requires_grad:
                    lora_trainable.append(p)
            for i in range(adapter.num_experts):
                a, b = adapter._get_ab(i)
                g = adapter.expert_gates[i]
                if a.requires_grad:
                    lora_trainable.extend([a, b, g])

        control_trainable = list(self.control_model.parameters())
        for p in control_trainable:
            p.requires_grad = True

        for p in self.model.diffusion_model.parameters():
            p.requires_grad = False

        for p in lora_trainable:
            p.requires_grad = True

        trainable = control_trainable + lora_trainable
        if len(trainable) == 0:
            raise RuntimeError(
                "No trainable parameters were found for control branch or LoRA adapters."
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
        if self.orth_constraint_mode == "soft":
            total_loss = loss + self.orth_lambda * orth_loss
        else:
            total_loss = loss

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
        if not self._warmup_done:
            self._warmup_batch_count += 1
            if self._warmup_batch_count >= self.warmup_novelty_batches:
                self._finish_task_warmup_and_grow()

    def on_train_start(self):
        self._start_task_warmup()

    def get_activation(self, name):
        def hook(model, input, output):
            base = model.module if isinstance(model, AdditiveLoRAAdapter) else model
            x = input[0].detach()
            if (isinstance(base, nn.Linear)
                    or isinstance(base, nn.modules.linear.NonDynamicallyQuantizableLinear)
                    or isinstance(base, MultiheadAttention)):

                input_channel = x.shape[-1]
                mat = x.reshape(-1, input_channel).t()

            elif isinstance(base, nn.Conv2d):
                input_channel = x.shape[1]
                pool_stride = self.conv_sample_pool_stride
                if pool_stride > 1 and x.shape[-1] >= pool_stride and x.shape[-2] >= pool_stride:
                    x = F.avg_pool2d(x, kernel_size=pool_stride, stride=pool_stride)
                unfolded = F.unfold(x, kernel_size=base.kernel_size, stride=base.stride, padding=base.padding)
                mat = unfolded.permute(1, 0, 2).reshape(unfolded.shape[1], -1)
                if mat.shape[1] > self.conv_sample_max_patches:
                    idx = torch.randperm(mat.shape[1], device=mat.device)[:self.conv_sample_max_patches]
                    mat = mat[:, idx]
            else:
                return

            if name not in self.act:
                self.act[name] = []
            self.act[name].append(mat)

        return hook

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            _ = self.log_images_test(batch)

    @torch.no_grad()
    def on_test_end(self):
        for value in self.hook_handle.values():
            value.remove()

        self.logger_val.info(
            "Skip test-end LoRA growth; warmup-growth is the only growth path."
        )
        self.task_id += 1
        self.max_check = 0.0

    @torch.no_grad()
    def on_test_start(self):
        self.hook_handle = {}