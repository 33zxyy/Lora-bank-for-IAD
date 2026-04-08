import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveLoRAAdapter(nn.Module):
    """Additive LoRA expert library for Conv2d/Linear with no-routing weighted fusion."""

    def __init__(self, module: nn.Module, rank: int = 4, alpha: float = 1.0,
                 max_experts: int = 32, router_hidden_dim: int = 64):
        super().__init__()
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            raise TypeError("AdditiveLoRAAdapter only supports nn.Linear and nn.Conv2d")

        self.module = module
        self.rank = rank
        self.alpha = alpha
        self.max_experts = max_experts
        self.experts = nn.ParameterList()
        self.expert_gates = nn.ParameterList()

        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
            router_in_dim = module.in_features
        else:
            in_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            out_dim = module.out_channels
            router_in_dim = module.in_channels

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.orth_mode = "soft"
        self.router_topk = 2
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, max_experts),
        )

    def add_expert(self, trainable: bool = True):
        device = self.module.weight.device
        dtype = self.module.weight.dtype
        # Lightning test/inference loops may run under torch.inference_mode().
        # Ensure newly appended trainable parameters are normal tensors.
        with torch.inference_mode(False):
            a = nn.Parameter(torch.zeros(self.rank, self.in_dim, device=device, dtype=dtype))
            b = nn.Parameter(torch.zeros(self.out_dim, self.rank, device=device, dtype=dtype))
            nn.init.zeros_(b)
            gate = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        a.requires_grad = trainable
        b.requires_grad = trainable
        gate.requires_grad = trainable

        self.experts.append(a)
        self.experts.append(b)
        self.expert_gates.append(gate)

    @property
    def num_experts(self):
        return len(self.expert_gates)

    def _get_ab(self, idx):
        return self.experts[2 * idx], self.experts[2 * idx + 1]

    def freeze_expert(self, idx):
        a, b = self._get_ab(idx)
        a.requires_grad = False
        b.requires_grad = False
        self.expert_gates[idx].requires_grad = False

    def unfreeze_expert(self, idx):
        a, b = self._get_ab(idx)
        a.requires_grad = True
        b.requires_grad = True
        self.expert_gates[idx].requires_grad = True

    def get_a(self, idx):
        a, _ = self._get_ab(idx)
        return a

    def orth_basis(self, upto: int = None):
        if self.num_experts == 0:
            return None
        if upto is None:
            upto = self.num_experts
        mats = [self.get_a(i).detach() for i in range(upto)]
        if len(mats) == 0:
            return None
        cat = torch.cat(mats, dim=0).t()  # [in_dim, sum_rank]
        q, _ = torch.linalg.qr(cat, mode='reduced')
        return q.t()  # [k, in_dim]

    def orthogonality_loss(self):
        if self.num_experts <= 1:
            return torch.tensor(0.0, device=self.module.weight.device)
        loss = torch.tensor(0.0, device=self.module.weight.device)
        q_list = []
        for i in range(self.num_experts):
            a = self.get_a(i)
            q, _ = torch.linalg.qr(a.t(), mode='reduced')
            q_list.append(q.t())  # [r, in_dim]
        for i in range(len(q_list)):
            for j in range(i + 1, len(q_list)):
                loss = loss + torch.norm(q_list[i] @ q_list[j].t(), p='fro') ** 2
        return loss

    def _project_to_old_subspace_complement(self, a, idx):
        if idx <= 0:
            return a
        u_prev = self.orth_basis(upto=idx)
        if u_prev is None or u_prev.numel() == 0:
            return a
        # A_eff = A_tilde (I - U^T U) = A_tilde - (A_tilde U^T) U
        return a - (a @ u_prev.t()) @ u_prev

    def novelty_energy(self, idx):
        if idx == 0:
            return 1.0
        target = self.get_a(idx).detach()
        u_prev = self.orth_basis(upto=idx)
        if u_prev is None or u_prev.numel() == 0:
            return 1.0
        proj = (target @ u_prev.t()) @ u_prev
        res = target - proj
        den = torch.norm(target, p='fro') ** 2 + 1e-8
        num = torch.norm(res, p='fro') ** 2
        return (num / den).item()

    def _routing_coeff(self, x):
        if self.num_experts == 0:
            return None
        if isinstance(self.module, nn.Conv2d):
            # x: [B, C, H, W] -> [B, C]
            feat = x.mean(dim=(2, 3))
        else:
            # x: [B, ..., C] -> [B, C]
            feat = x.reshape(-1, x.shape[-1])
        feat = feat.detach()
        logits = self.router(feat)[:, :self.num_experts]
        gate_bias = torch.stack([g for g in self.expert_gates], dim=0).unsqueeze(0)
        logits = logits + gate_bias
        topk = self.num_experts if self.router_topk is None else max(int(self.router_topk), 1)
        topk = min(topk, self.num_experts)
        topv, topi = torch.topk(logits, k=topk, dim=-1)
        coeff_sparse = torch.zeros_like(logits)
        coeff_sparse.scatter_(-1, topi, torch.softmax(topv, dim=-1))
        return coeff_sparse

    def _delta_output(self, x, routing_coeff, out_shape=None):
        if self.num_experts == 0 or routing_coeff is None:
            if out_shape is None:
                return None
            return torch.zeros(out_shape, device=x.device, dtype=x.dtype)

        if isinstance(self.module, nn.Linear):
            x_flat = x.reshape(-1, x.shape[-1])
            delta_flat = torch.zeros(
                x_flat.shape[0], self.out_dim, device=x.device, dtype=x.dtype
            )
            for i in range(self.num_experts):
                coeff = routing_coeff[:, i:i + 1]
                if torch.count_nonzero(coeff).item() == 0:
                    continue
                a, b = self._get_ab(i)
                if self.orth_mode == "hard":
                    a = self._project_to_old_subspace_complement(a, i)
                d = (x_flat @ a.t()) @ b.t()
                delta_flat = delta_flat + coeff * d
            return self.alpha * delta_flat.view(*x.shape[:-1], self.out_dim)

        bsz = x.shape[0]
        x_unfold = F.unfold(
            x,
            kernel_size=self.module.kernel_size,
            dilation=self.module.dilation,
            padding=self.module.padding,
            stride=self.module.stride
        ).transpose(1, 2)
        delta_tokens = torch.zeros(
            bsz, x_unfold.shape[1], self.out_dim, device=x.device, dtype=x.dtype
        )
        for i in range(self.num_experts):
            coeff = routing_coeff[:, i].view(bsz, 1, 1)
            if torch.count_nonzero(coeff).item() == 0:
                continue
            a, b = self._get_ab(i)
            if self.orth_mode == "hard":
                a = self._project_to_old_subspace_complement(a, i)
            d = (x_unfold @ a.t()) @ b.t()
            delta_tokens = delta_tokens + coeff * d

        if out_shape is None:
            raise ValueError("out_shape must be provided for Conv2d delta reconstruction.")
        h_out, w_out = out_shape[-2], out_shape[-1]
        return self.alpha * delta_tokens.transpose(1, 2).reshape(bsz, self.out_dim, h_out, w_out)

    def merged_delta(self, routing_coeff=None):
        if self.num_experts == 0:
            return torch.zeros_like(self.module.weight)
        delta = torch.zeros_like(self.module.weight)
        if routing_coeff is None:
            gate_values = torch.stack([g for g in self.expert_gates], dim=0)
            topk = self.num_experts if self.router_topk is None else max(int(self.router_topk), 1)
            topk = min(topk, self.num_experts)
            selected = torch.topk(gate_values, k=topk, dim=0).indices.tolist()
            coeffs = torch.softmax(gate_values[selected], dim=0)
        else:
            topk = self.num_experts if self.router_topk is None else max(int(self.router_topk), 1)
            topk = min(topk, self.num_experts)
            selected = torch.topk(routing_coeff, k=topk, dim=0).indices.tolist()
            coeffs = routing_coeff[selected]
            coeffs = coeffs / (coeffs.sum() + 1e-8)
        for coeff, i in zip(coeffs, selected):
            a, b = self._get_ab(i)
            if self.orth_mode == "hard":
                a = self._project_to_old_subspace_complement(a, i)
            d = (b @ a).view_as(self.module.weight)
            delta = delta + coeff * d
        return self.alpha * delta

    def forward(self, x):
        routing_coeff = self._routing_coeff(x)
        base = self.module(x)
        delta = self._delta_output(x, routing_coeff, out_shape=base.shape)
        return base + delta

    @torch.no_grad()
    def fuse_into_weight(self):
        self.module.weight.data.copy_(self.module.weight.data + self.merged_delta().data)