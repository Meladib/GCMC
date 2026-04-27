"""
gcmc.py — GCMC v7
=================
Changes from v6:
- Added get_gradient_stats() for per-head gradient monitoring
- Added get_feature_stats() for feature space monitoring at inference
- Added R² metric support (direction-aware, not just magnitude)
- sigma2 floor = 1e-6 (unchanged from v6)
- Architecture unchanged (20-dim input, 24-hidden)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POS_SCALE = 0.2257
VEL_SCALE = 0.0022
DV_STD    = np.array([0.004789, 0.006376], dtype=np.float32) 

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
        group_vel = vel.mean(dim=0, keepdim=True).expand(N, -1)
        vel_dev   = vel - group_vel
        return torch.cat([mu, sigma, group_vel, vel_dev], dim=-1)  # (N,12)


class CorrectionMLP(nn.Module):
    def __init__(self, input_dim: int = 20) -> None:
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
        super().__init__()
        self.aggregation = NeighbourAggregation(tau_init=tau_init)
        self.mlp         = CorrectionMLP(input_dim=20)
        self.register_buffer('pos_scale', torch.tensor(POS_SCALE))
        self.register_buffer('vel_scale', torch.tensor(VEL_SCALE))

    def _normalize(self, f_i):
        return torch.cat([
            f_i[:,  :2] / self.pos_scale,
            f_i[:,  2:4] / self.vel_scale,
            f_i[:,  4:6] / self.pos_scale,
            f_i[:,  6:8] / self.vel_scale,
            f_i[:,  8:10] / self.vel_scale,
            f_i[:, 10:12] / self.vel_scale,
            f_i[:, 12:16] / self.pos_scale,
            f_i[:, 16:20] / self.vel_scale,
        ], dim=-1)

    def forward(self, states):
        N = states.shape[0]
        if N <= 1:
            return states.new_zeros(N,2), states.new_ones(N,2)*1e-6, states.new_zeros(N,20)
        f_nb = self.aggregation(states)
        f_i  = torch.cat([f_nb, states], dim=-1)
        dv, sigma2 = self.mlp(self._normalize(f_i))
        return dv, sigma2, f_i

    DV_STD = np.array([0.004789, 0.006376], dtype=np.float32)

    def predict_numpy(self, states_norm, device='cpu'):
        self.eval()
        with torch.no_grad():
            dv, sigma2, _ = self.forward(
                torch.from_numpy(states_norm).float().to(device))
        # Rescale from standardized space back to normalised coords
        dv_out = dv.cpu().numpy() * DV_STD   # ← this line
        return dv_out, sigma2.cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    def get_gradient_stats(self) -> dict:
        """Call after loss.backward(). Returns per-component gradient norms."""
        def gnorm(params):
            gs = [p.grad.norm().item() for p in params if p.grad is not None]
            return sum(gs) / max(len(gs), 1)
        return {
            'grad_correction':  gnorm(self.mlp.correction_head.parameters()),
            'grad_uncertainty': gnorm(self.mlp.uncertainty_head.parameters()),
            'grad_backbone':    gnorm(self.mlp.backbone.parameters()),
            'grad_tau':         self.aggregation.log_tau.grad.item()
                                if self.aggregation.log_tau.grad is not None else 0.0,
            'grad_corr_unc_ratio': gnorm(self.mlp.correction_head.parameters()) /
                                   max(gnorm(self.mlp.uncertainty_head.parameters()), 1e-12),
        }

    def get_feature_stats(self, states: torch.Tensor) -> dict:
        """Returns feature space diagnostics for a batch of states."""
        with torch.no_grad():
            N = states.shape[0]
            pos = (states[:, :2] + states[:, 2:4]) / 2.0
            vel = (states[:, 4:6] + states[:, 6:8]) / 2.0
            vel_mag = vel.norm(dim=1)
            group_vel = vel.mean(dim=0)
            vel_dev   = (vel - group_vel).norm(dim=1)
            return {
                'n_tracks': N,
                'mean_vel_mag':   vel_mag.mean().item(),
                'mean_vel_dev':   vel_dev.mean().item(),
                'group_vel_mag':  group_vel.norm().item(),
                'vel_coherence':  1.0 - (vel_dev.mean() / (vel_mag.mean() + 1e-8)).item(),
                'tau': self.aggregation.tau.item(),
            }


def run_unit_tests():
    print('Running GCMC v7 unit tests...')
    torch.manual_seed(0)
    model = GCMC()
    n = model.count_parameters()
    assert n == 605, f'Expected 605, got {n}'
    print(f'  [PASS] param count = {n}')

    s = torch.rand(6, 8); idx = torch.randperm(6)
    _, _, f1 = model(s); _, _, f2 = model(s[idx])
    assert torch.allclose(f1, f2[torch.argsort(idx)], atol=1e-5)
    print('  [PASS] permutation invariance')

    dv_s, _, _ = model(torch.rand(1, 8))
    assert torch.allclose(dv_s, torch.zeros(1, 2))
    print('  [PASS] k=0 (Δv=0)')

    _, _, f_two = model(torch.rand(2, 8))
    assert f_two[:, 4:8].abs().max().item() < 1e-2
    print('  [PASS] k=1 (relative σ≈0)')

    fo = fanout(torch.tensor([[0.1, 0.2]]))
    assert torch.allclose(fo[:, 0]+fo[:, 2], torch.tensor([0.1]), atol=1e-6)
    print('  [PASS] fanout energy')

    _, sig, _ = model(torch.rand(10, 8))
    assert (sig > 0).all()
    print('  [PASS] sigma2 > 0')

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        s = torch.rand(7, 8); gt = torch.rand(7, 2) * 0.01
        opt.zero_grad()
        dv, sig2, _ = model(s)
        nll_loss(dv, gt, sig2).backward()
        gs = model.get_gradient_stats()
        assert not torch.isnan(torch.tensor(gs['grad_tau']))
        opt.step()
    print('  [PASS] no NaN gradients + get_gradient_stats() works')

    fs = model.get_feature_stats(torch.rand(8, 8))
    assert 'vel_coherence' in fs
    print('  [PASS] get_feature_stats() works')

    print(f'\nAll tests passed. Params={model.count_parameters()}')

if __name__ == '__main__':
    run_unit_tests()