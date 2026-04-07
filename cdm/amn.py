
import torch
import torch.nn as nn

from scipy.ndimage import gaussian_filter
from utils.util import cal_anomaly_map, log_local, create_logger
from utils.eval_helper import dump, log_metrics, merge_together, performances, save_metrics
from cdm.param import no_trained_para, control_trained_para, contains_any, sub_


import os

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from cdm.vit import *

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)

        for module in self.input_blocks:
            if control is not None:
                h = module(h, emb, context)
                h += control
                control = None
            else:
                h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)


class AMN(nn.Module):
    def __init__(
            self,
            image_size,
            model_channels,
            hint_channels,
            dims=2,
            pos_embed_type='sine',
            neighbor_size=(7, 7),
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=1025,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        # Explicit marker: AMN/control branch should stay LoRA-free.
        self.disable_lora = True

        self.model_channels = model_channels

        self.neighbor_size = neighbor_size

        self.img_size = (image_size, image_size)

        self.input_hint_block = nn.Sequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1, bias=False),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1, bias=False),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2, bias=False),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1, bias=False),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2, bias=False),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1, bias=False),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2, bias=False),
            nn.SiLU(),
            conv_nd(dims, 256, model_channels, 3, padding=1, bias=False)
        )

        self.pos_embed = build_position_embedding(
            pos_embed_type, [image_size, image_size], model_channels
        )

        encoder_layer = TransformerEncoderLayer(
            model_channels, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(model_channels) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, hint, re=True):

        # guided_hint = self.input_hint_block(hint)
        hint_feature = self.input_hint_block(hint)

        feature_tokens = rearrange(
            hint_feature, "b c h w -> (h w) b c"
        )

        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C

        _, batch_size, _ = feature_tokens.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )

        if self.neighbor_size:
            mask = self.generate_mask(
                self.img_size, self.neighbor_size
            )
        else:
            mask = None

        output_encoder = self.encoder(
            feature_tokens, mask=mask, pos=pos_embed
        )  # B X (H X W) x C

        feature_rec = rearrange(
            output_encoder, "(h w) b c -> b c h w", h=self.img_size[0]
        )  # B x C X H x W

        if re:
            return feature_rec
        else:
            return hint_feature