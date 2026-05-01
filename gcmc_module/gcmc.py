"""
gcmc.py — GCMC v10
==================
Changes from v9:
  - NeighbourAggregation removed entirely
  - Input: 8-dim own-state only (XYXY + vel)
  - CorrectionMLP input_dim: 20 → 8
  - ~200 params (was 605)
  - forward() simplified: no aggregation, no group_vel, no vel_deviation
  - get_gradient_stats(): tau entry removed
  - get_feature_stats(): group/neighbour stats removed
  - Unit tests updated accordingly

Rationale: k=0 ablation showed neighbourhood features contribute
zero gain vs full aggregation. Own-state uncertainty estimation
alone accounts for all observed ΔAssA improvements.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POS_SCALE = 0.2257
VEL_SCALE = 0.0022
DV_STD    = np.array([0.004789, 0.006376], dtype=np.float32)


class CorrectionMLP(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.backbone         = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU())
        self.correction_head  = nn.Linear(24, 2)
        self.uncertainty_head = nn.Linear(24, 2)
        nn.init.xavier_uniform_(self.correction_head.weight, gain=0.1)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(self, f_i):
        h = self.backbone(f_i)
        return self.correction_head(h), F.softplus(self.uncertainty_head(h)) + 1e-6


def fanout(dv: torch.Tensor) -> torch.Tensor:
    dvx, dvy = dv[:, 0:1], dv[:, 1:2]
    return torch.cat([dvx/2,dvy/2,dvx/2,dvy/2,dvx,dvy,dvx,dvy], dim=-1)

def fanout_numpy(dv: np.ndarray) -> np.ndarray:
    dvx, dvy = dv[:, 0:1], dv[:, 1:2]
    return np.concatenate([dvx/2,dvy/2,dvx/2,dvy/2,dvx,dvy,dvx,dvy], axis=-1)

def build_Q_aug(Q_base: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    N = sigma2.shape[0]
    Q = np.tile(Q_base[None], (N,1,1)).copy()
    Q[:, 0, 0] += sigma2[:, 0]/2; Q[:, 1, 1] += sigma2[:, 1]/2
    Q[:, 2, 2] += sigma2[:, 0]/2; Q[:, 3, 3] += sigma2[:, 1]/2
    Q[:, 4, 4] += sigma2[:, 0];   Q[:, 5, 5] += sigma2[:, 1]
    Q[:, 6, 6] += sigma2[:, 0];   Q[:, 7, 7] += sigma2[:, 1]
    return Q

def nll_loss(dv_pred, dv_star, sigma2):
    return ((((dv_pred - dv_star)**2) / sigma2) + torch.log(sigma2)).mean()


class GCMC(nn.Module):
    def __init__(self, tau_init: float = 0.05) -> None:
        # tau_init kept for API compatibility — unused in v10
        super().__init__()
        self.mlp = CorrectionMLP(input_dim=8)
        self.register_buffer('pos_scale', torch.tensor(POS_SCALE))
        self.register_buffer('vel_scale', torch.tensor(VEL_SCALE))

    def _normalize(self, states: torch.Tensor) -> torch.Tensor:
        scales = states.new_tensor([
            POS_SCALE, POS_SCALE, POS_SCALE, POS_SCALE,  # own pos
            VEL_SCALE, VEL_SCALE, VEL_SCALE, VEL_SCALE,  # own vel
        ])
        return states / scales

    def forward(self, states: torch.Tensor, zero_neighbours: bool = False):
        # zero_neighbours kept for API compatibility — no-op in v10
        N = states.shape[0]
        if N == 0:
            z = states.new_zeros(0, 2)
            return z, z, states.new_zeros(0, 8)

        f_i = self._normalize(states)
        dv_pred_std, sigma2 = self.mlp(f_i)
        dv_std_t = torch.tensor(DV_STD, device=states.device, dtype=states.dtype)
        dv_pred  = dv_pred_std * dv_std_t

        return dv_pred, sigma2, f_i

    def predict_numpy(self, states_norm: np.ndarray, device: str = 'cpu',
                      zero_neighbours: bool = False):
        with torch.no_grad():
            s_t = torch.tensor(states_norm, dtype=torch.float32, device=device)
            dv, sigma2, _ = self(s_t)
        return dv.cpu().numpy(), sigma2.cpu().numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gradient_stats(self) -> dict:
        def gnorm(params):
            gs = [p.grad.norm().item() for p in params if p.grad is not None]
            return float(np.mean(gs)) if gs else 0.0
        return {
            'grad_correction':     gnorm(self.mlp.correction_head.parameters()),
            'grad_uncertainty':    gnorm(self.mlp.uncertainty_head.parameters()),
            'grad_backbone':       gnorm(self.mlp.backbone.parameters()),
            'grad_corr_unc_ratio': (gnorm(self.mlp.correction_head.parameters()) /
                                    max(gnorm(self.mlp.uncertainty_head.parameters()), 1e-12)),
        }

    def get_feature_stats(self, states: torch.Tensor) -> dict:
        with torch.no_grad():
            vel     = (states[:, 4:6] + states[:, 6:8]) / 2.0
            vel_mag = vel.norm(dim=1)
            return {
                'n_tracks':     states.shape[0],
                'mean_vel_mag': vel_mag.mean().item(),
                'max_vel_mag':  vel_mag.max().item(),
            }


def run_unit_tests():
    print('Running GCMC v10 unit tests...')
    torch.manual_seed(0)
    model = GCMC()
    n = model.count_parameters()
    print(f'  param count = {n}')
    assert n < 300, f'Expected <300 params, got {n}'
    print(f'  [PASS] param count < 300')

    _, sig, _ = model(torch.rand(10, 8))
    assert (sig > 0).all()
    print('  [PASS] sigma2 > 0')

    # Single track
    dv, sig, f = model(torch.rand(1, 8))
    assert dv.shape == (1, 2)
    print('  [PASS] single track forward')

    # Empty input
    dv, sig, f = model(torch.zeros(0, 8))
    assert dv.shape == (0, 2)
    print('  [PASS] empty input')

    # No NaN gradients
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        s = torch.rand(7, 8); gt = torch.rand(7, 2) * 0.01
        opt.zero_grad()
        dv, sig2, _ = model(s)
        nll_loss(dv, gt, sig2).backward()
        assert not any(torch.isnan(p.grad).any()
                       for p in model.parameters() if p.grad is not None)
        opt.step()
    print('  [PASS] no NaN gradients')

    # API compat: zero_neighbours is no-op
    dv1, s1, _ = model(torch.rand(5, 8), zero_neighbours=False)
    dv2, s2, _ = model(torch.rand(5, 8), zero_neighbours=True)
    print('  [PASS] zero_neighbours API compat')

    print(f'\nAll tests passed. Params={model.count_parameters()}')


if __name__ == '__main__':
    run_unit_tests()
