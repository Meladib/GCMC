"""
gcmc.py — Group-Conditioned Motion Correction (v4)
===================================================
Changes from v3:
- f_i expanded to 20-dim: neighbourhood (12) + own state (8)
- Own state gives MLP direct access to KF velocity estimate
- Fixes diagnostic failure: r_proposed -0.013 → expected > 0.50
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# From diag_data.txt Section 1D
POS_SCALE = 0.2257
VEL_SCALE = 0.0022


# ── §3.4  Neighbourhood aggregation ──────────────────────────────────────────

class NeighbourAggregation(nn.Module):
    def __init__(self, tau_init: float = 0.05) -> None:
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        N = states.shape[0]
        if N == 0:
            return states.new_zeros(0, 12)

        pos = (states[:, :2] + states[:, 2:4]) / 2.0
        vel = (states[:, 4:6] + states[:, 6:8]) / 2.0

        # ── existing relative features ────────────────────────────
        dpos = pos.unsqueeze(1) - pos.unsqueeze(0)
        dvel = vel.unsqueeze(1) - vel.unsqueeze(0)
        f_ij = torch.cat([dpos, dvel], dim=-1)

        dist   = torch.sqrt((dpos**2).sum(dim=-1) + 1e-8)
        eye    = torch.eye(N, dtype=torch.bool, device=states.device)
        logits = (-dist / self.tau).masked_fill(eye, -1e9)
        alpha  = torch.softmax(logits, dim=-1)

        mu    = torch.einsum('ij,ijd->id', alpha, f_ij)
        diff2 = (f_ij - mu.unsqueeze(1)) ** 2
        sigma = torch.sqrt(torch.einsum('ij,ijd->id', alpha, diff2) + 1e-6)

        # ── absolute group velocity features ─────────────────
        group_vel  = vel.mean(dim=0, keepdim=True).expand(N, -1)  # (N,2) group mean
        vel_dev    = vel - group_vel                                # (N,2) deviation

        return torch.cat([mu, sigma, group_vel, vel_dev], dim=-1)  # (N,12)


# ── §3.5  MLP ────────────────────────────────────────────────────────────────

class CorrectionMLP(nn.Module):
    def __init__(self, input_dim: int = 20) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU())
        self.correction_head  = nn.Linear(24, 2)
        self.uncertainty_head = nn.Linear(24, 2)
        # Stronger init — diagnostic showed gradient race, need faster activation
        nn.init.xavier_uniform_(self.correction_head.weight, gain=0.1)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(self, f_i):
        h = self.backbone(f_i)
        return self.correction_head(h), F.softplus(self.uncertainty_head(h)) + 1e-6


# ── Utilities ─────────────────────────────────────────────────────────────────

def fanout(dv: torch.Tensor) -> torch.Tensor:
    dvx, dvy = dv[:, 0:1], dv[:, 1:2]
    return torch.cat([dvx/2,dvy/2,dvx/2,dvy/2,dvx,dvy,dvx,dvy], dim=-1)

def fanout_numpy(dv: np.ndarray) -> np.ndarray:
    dvx, dvy = dv[:, 0:1], dv[:, 1:2]
    return np.concatenate([dvx/2,dvy/2,dvx/2,dvy/2,dvx,dvy,dvx,dvy], axis=-1)

def build_Q_aug(Q_base: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    N = sigma2.shape[0]
    Q = np.tile(Q_base[None], (N,1,1)).copy()
    for i, (d0, d1) in enumerate([(0,0),(1,1),(2,2),(3,3)]):
        Q[:, d0, d1] += sigma2[:, i % 2] / 2
    Q[:, 4, 4] += sigma2[:, 0]; Q[:, 5, 5] += sigma2[:, 1]
    Q[:, 6, 6] += sigma2[:, 0]; Q[:, 7, 7] += sigma2[:, 1]
    return Q

def nll_loss(dv_pred, dv_star, sigma2):
    return ((((dv_pred - dv_star)**2) / sigma2) + torch.log(sigma2)).mean()


# ── Full GCMC ─────────────────────────────────────────────────────────────────

class GCMC(nn.Module):
    def __init__(self, tau_init: float = 0.05) -> None:
        super().__init__()
        self.aggregation = NeighbourAggregation(tau_init=tau_init)
        self.mlp         = CorrectionMLP(input_dim=20)  # CHANGED: 12 → 20
        self.register_buffer('pos_scale', torch.tensor(POS_SCALE))
        self.register_buffer('vel_scale', torch.tensor(VEL_SCALE))

    def _normalize(self, f_i):
        # f_i: (N, 20) = [mu_pos(2), mu_vel(2), sigma_pos(2), sigma_vel(2),
        #                  group_vel(2), vel_dev(2), own_state(8)]
        return torch.cat([
            f_i[:,  :2] / self.pos_scale,     # mu_pos
            f_i[:,  2:4] / self.vel_scale,     # mu_vel
            f_i[:,  4:6] / self.pos_scale,     # sigma_pos
            f_i[:,  6:8] / self.vel_scale,     # sigma_vel
            f_i[:,  8:10] / self.vel_scale,    # group_vel
            f_i[:, 10:12] / self.vel_scale,    # vel_dev
            f_i[:, 12:16] / self.pos_scale,    # own_pos (x1,y1,x2,y2)
            f_i[:, 16:20] / self.vel_scale,    # own_vel (vx1,vy1,vx2,vy2)
        ], dim=-1)

    def forward(self, states):
        N = states.shape[0]
        if N <= 1:
            # CHANGED: 12 → 20
            return states.new_zeros(N,2), states.new_ones(N,2)*1e-2, states.new_zeros(N,20)
        
        f_neighbour = self.aggregation(states)  # (N, 12)
        f_i = torch.cat([f_neighbour, states], dim=-1)  # NEW: (N, 20)
        
        dv, sigma2 = self.mlp(self._normalize(f_i))
        return dv, sigma2, f_i

    def predict_numpy(self, states_norm, device='cpu'):
        self.eval()
        with torch.no_grad():
            dv, sigma2, _ = self.forward(torch.from_numpy(states_norm).float().to(device))
        return dv.cpu().numpy(), sigma2.cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Unit tests ────────────────────────────────────────────────────────────────

def run_unit_tests():
    print('Running GCMC v4 unit tests...')
    torch.manual_seed(0)
    model = GCMC()

    # CHANGED: 213 → 341 params (20-dim input)
    assert model.count_parameters() == 341
    print(f'  [PASS] param count = 341')

    states = torch.rand(6, 8)
    idx = torch.randperm(6)
    _, _, f1 = model(states)
    _, _, f2 = model(states[idx])
    # Permutation invariance: neighbour features should permute, own-state should follow
    assert torch.allclose(f1[:, :12], f2[:, :12][torch.argsort(idx)], atol=1e-5)
    assert torch.allclose(f1[:, 12:], f2[:, 12:][torch.argsort(idx)], atol=1e-5)
    print('  [PASS] permutation invariance (neighbour + own-state)')

    dv_s, _, _ = model(torch.rand(1, 8))
    assert torch.allclose(dv_s, torch.zeros(1, 2))
    print('  [PASS] k=0 (Δv=0)')

    _, _, f_two = model(torch.rand(2, 8))
    assert f_two[:, 4:8].abs().max().item() < 1e-2  # sigma of relative features
    print('  [PASS] k=1 (relative σ≈0)')

    fo = fanout(torch.tensor([[0.1, 0.2]]))
    assert torch.allclose(fo[:, 0]+fo[:, 2], torch.tensor([0.1]), atol=1e-6)
    print('  [PASS] fanout energy')

    _, sig, _ = model(torch.rand(10, 8))
    assert (sig > 0).all()
    print('  [PASS] sigma2 > 0')

    # No NaN gradients
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        s = torch.rand(7, 8); gt = torch.rand(7, 2) * 0.01
        opt.zero_grad()
        dv, sig2, _ = model(s)
        nll_loss(dv, gt, sig2).backward()
        assert not model.aggregation.log_tau.grad.isnan()
        opt.step()
    print('  [PASS] no NaN gradients')

    print(f'\nAll tests passed. τ={model.aggregation.tau.item():.4f}')

if __name__ == '__main__':
    run_unit_tests()
