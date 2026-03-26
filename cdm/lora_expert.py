import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveLoRAAdapter(nn.Module):
    """Additive LoRA expert library for Conv2d/Linear with no-routing weighted fusion."""

    def __init__(self, module: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            raise TypeError("AdditiveLoRAAdapter only supports nn.Linear and nn.Conv2d")

        self.module = module
        self.rank = rank
        self.alpha = alpha
        self.experts = nn.ParameterList()
        self.expert_gates = nn.ParameterList()

        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
        else:
            in_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            out_dim = module.out_channels

        self.in_dim = in_dim
        self.out_dim = out_dim

    def add_expert(self, trainable: bool = True):
        with torch.inference_mode(False):
            a=nn.Parameter(torch.zerros(self.rank,self.in_dim))
            b=nn.Parameter(torch.zeros(self.out_dim,self.rank))
            nn.init.kaiming_uniform_(a,a=5 ** 0.5)
            nn.init.zeros_(b)
            gate=nn.Parameter(torch.tensor(1.0))
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

    def merged_delta(self):
        if self.num_experts == 0:
            return torch.zeros_like(self.module.weight)
        delta = torch.zeros_like(self.module.weight)
        for i in range(self.num_experts):
            a, b = self._get_ab(i)
            d = (b @ a).view_as(self.module.weight)
            delta = delta + self.expert_gates[i] * d
        return self.alpha * delta

    def forward(self, x):
        w_eff = self.module.weight + self.merged_delta()
        if isinstance(self.module, nn.Linear):
            return F.linear(x, w_eff, self.module.bias)
        return F.conv2d(
            x,
            w_eff,
            self.module.bias,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups,
        )

    @torch.no_grad()
    def fuse_into_weight(self):
        self.module.weight.data.copy_(self.module.weight.data + self.merged_delta().data)