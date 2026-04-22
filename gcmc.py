"""
gcmc.py — Group-Conditioned Motion Correction Module
=====================================================
Phase 3 implementation — §3.4 + §3.5 + §3.6

State convention (XYXY 8-dim):
    [x1, y1, x2, y2, vx1, vy1, vx2, vy2]  — all values normalised to [0,1]

Output:
    dv    : (N, 2)  center-space velocity correction  Δv ∈ ℝ²
    sigma2: (N, 2)  per-track heteroscedastic variance σ² ∈ ℝ²₊
    fanout: (N, 8)  Δv expanded to full 8-dim state correction
    Q_aug : (N, 8, 8) augmented process noise matrices
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# §3.4  Neighbourhood aggregation
# ─────────────────────────────────────────────────────────────────────────────

class NeighbourAggregation(nn.Module):
    """Distance-weighted softmax aggregation → dual-stats (μ, σ) ∈ ℝ⁸.

    Args:
        tau_init: Initial value of learnable temperature τ (in normalised
                  coord units). Default 0.05 ≈ 50px / 1920.
    """

    def __init__(self, tau_init: float = 0.05) -> None:
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (N, 8) normalised XYXY states.

        Returns:
            f_i: (N, 8) — [μ ‖ σ] aggregated neighbourhood features.
        """
        N = states.shape[0]

        if N == 0:
            return states.new_zeros(0, 8)

        # ── Center positions and velocities ──────────────────────────────
        # pos: (N, 2)  vel: (N, 2)
        pos = (states[:, :2] + states[:, 2:4]) / 2.0   # center xy
        vel = (states[:, 4:6] + states[:, 6:8]) / 2.0  # center vxy

        # ── Pairwise relative features f_ij ∈ ℝ⁴ ─────────────────────────
        # dpos: (N, N, 2)  dvel: (N, N, 2)
        dpos = pos.unsqueeze(1) - pos.unsqueeze(0)   # (N, N, 2)
        dvel = vel.unsqueeze(1) - vel.unsqueeze(0)   # (N, N, 2)
        f_ij = torch.cat([dpos, dvel], dim=-1)        # (N, N, 4)

        # ── Spatial distances ─────────────────────────────────────────────
        dist = torch.norm(dpos, dim=-1)               # (N, N)

        # Mask self-connections
        eye_mask = torch.eye(N, dtype=torch.bool, device=states.device)
        dist = dist.masked_fill(eye_mask, float('inf'))

        # ── Distance-weighted softmax attention ───────────────────────────
        alpha = torch.softmax(-dist / self.tau, dim=-1)  # (N, N)

        # ── Weighted mean and std ─────────────────────────────────────────
        mu    = torch.einsum('ij,ijd->id', alpha, f_ij)                    # (N, 4)
        diff2 = (f_ij - mu.unsqueeze(1)) ** 2
        sigma = torch.sqrt(
            torch.einsum('ij,ijd->id', alpha, diff2) + 1e-6
        )                                                                    # (N, 4)

        return torch.cat([mu, sigma], dim=-1)                               # (N, 8)


# ─────────────────────────────────────────────────────────────────────────────
# §3.5  MLP: correction + uncertainty heads
# ─────────────────────────────────────────────────────────────────────────────

class CorrectionMLP(nn.Module):
    """Shared backbone → dual output heads.

    Input:  f_i ∈ ℝ⁸  (aggregated neighbourhood features)
    Output: Δv ∈ ℝ²,  log σ² ∈ ℝ²
    Total params: 213 (incl. τ)
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone         = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.correction_head  = nn.Linear(16, 2)
        self.uncertainty_head = nn.Linear(16, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init correction head → starts as identity (no correction)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        # Uncertainty head: small positive init → σ² ≈ softplus(0) ≈ 0.69
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(
        self, f_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_i: (N, 8)

        Returns:
            dv:     (N, 2)  velocity correction Δv
            sigma2: (N, 2)  uncertainty σ² (strictly positive)
        """
        h      = self.backbone(f_i)                              # (N, 16)
        dv     = self.correction_head(h)                         # (N, 2)
        log_s2 = self.uncertainty_head(h)                        # (N, 2)
        sigma2 = F.softplus(log_s2) + 1e-2                     # (N, 2) > 0
        return dv, sigma2


# ─────────────────────────────────────────────────────────────────────────────
# §3.5  Fan-out: center Δv → 8-dim state correction
# ─────────────────────────────────────────────────────────────────────────────

def fanout(dv: torch.Tensor) -> torch.Tensor:
    """Expand center-space Δv ∈ ℝ² to full 8-dim XYXY correction.

    Position dims (0-3) get Δv/2 (symmetric split, Δt=1).
    Velocity dims (4-7) get Δv directly.

    Args:
        dv: (N, 2) — [Δvx, Δvy]

    Returns:
        correction: (N, 8)
    """
    dvx = dv[:, 0:1]   # (N, 1)
    dvy = dv[:, 1:2]   # (N, 1)
    return torch.cat([
        dvx / 2, dvy / 2,   # x1, y1
        dvx / 2, dvy / 2,   # x2, y2
        dvx,     dvy,        # vx1, vy1
        dvx,     dvy,        # vx2, vy2
    ], dim=-1)               # (N, 8)


def fanout_numpy(dv: np.ndarray) -> np.ndarray:
    """Numpy version of fanout for inference integration."""
    dvx = dv[:, 0:1]
    dvy = dv[:, 1:2]
    return np.concatenate([
        dvx/2, dvy/2, dvx/2, dvy/2,
        dvx,   dvy,   dvx,   dvy,
    ], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# §3.5  Q_aug: augmented process noise
# ─────────────────────────────────────────────────────────────────────────────

def build_Q_aug(
    Q_base: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Augment per-track process noise with predicted uncertainty.

    Args:
        Q_base:  (8, 8) base process noise matrix.
        sigma2:  (N, 2) predicted [σ²_vx, σ²_vy] per track.

    Returns:
        Q_aug: (N, 8, 8) per-track augmented Q matrices.
    """
    N = sigma2.shape[0]
    Q_aug = np.tile(Q_base[None], (N, 1, 1)).copy()  # (N, 8, 8)

    # Position dims: σ²/2 (Δt=1, symmetric corner split)
    Q_aug[:, 0, 0] += sigma2[:, 0] / 2   # x1
    Q_aug[:, 1, 1] += sigma2[:, 1] / 2   # y1
    Q_aug[:, 2, 2] += sigma2[:, 0] / 2   # x2
    Q_aug[:, 3, 3] += sigma2[:, 1] / 2   # y2
    # Velocity dims: σ² directly
    Q_aug[:, 4, 4] += sigma2[:, 0]        # vx1
    Q_aug[:, 5, 5] += sigma2[:, 1]        # vy1
    Q_aug[:, 6, 6] += sigma2[:, 0]        # vx2
    Q_aug[:, 7, 7] += sigma2[:, 1]        # vy2

    return Q_aug


# ─────────────────────────────────────────────────────────────────────────────
# §3.6  NLL training loss
# ─────────────────────────────────────────────────────────────────────────────

def nll_loss(
    dv_pred:  torch.Tensor,
    dv_star:  torch.Tensor,
    sigma2:   torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic NLL loss.

    L = Σᵢ [ ‖Δvᵢ − Δvᵢ*‖² / σᵢ²  +  log σᵢ² ]

    Args:
        dv_pred: (N, 2) predicted corrections
        dv_star: (N, 2) ground-truth residuals
        sigma2:  (N, 2) predicted variances

    Returns:
        Scalar loss.
    """
    sq_err = (dv_pred - dv_star) ** 2           # (N, 2)
    loss   = (sq_err / sigma2) + torch.log(sigma2)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full GCMC module (aggregation + MLP)
# ─────────────────────────────────────────────────────────────────────────────

class GCMC(nn.Module):
    """Full GCMC module: neighbourhood encoding + correction + uncertainty.

    Args:
        tau_init: Initial temperature for spatial attention (normalised coords).
                  Default 0.05 ≈ 50px/1920.

    Forward input:
        states: (N, 8) normalised XYXY states of all active tracks.

    Forward output:
        dv:      (N, 2)  center-space correction
        sigma2:  (N, 2)  uncertainty
        f_i:     (N, 8)  aggregated features (for inspection/ablation)
    """

    def __init__(self, tau_init: float = 0.05) -> None:
        super().__init__()
        self.aggregation = NeighbourAggregation(tau_init=tau_init)
        self.mlp         = CorrectionMLP()

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = states.shape[0]

        # Edge case: 0 or 1 track — no neighbours, no correction
        if N <= 1:
            dv     = states.new_zeros(N, 2)
            sigma2 = states.new_ones(N, 2) * 1e-6
            f_i    = states.new_zeros(N, 8)
            return dv, sigma2, f_i

        f_i          = self.aggregation(states)    # (N, 8)
        dv, sigma2   = self.mlp(f_i)               # (N,2), (N,2)
        return dv, sigma2, f_i

    def predict_numpy(
        self,
        states_norm: np.ndarray,
        device: str = 'cpu',
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference-time helper: numpy in → numpy out.

        Args:
            states_norm: (N, 8) normalised XYXY states.
            device: torch device string.

        Returns:
            dv_norm:  (N, 2) correction in normalised coords
            sigma2:   (N, 2) uncertainty
        """
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(states_norm).float().to(device)
            dv, sigma2, _ = self.forward(t)
        return dv.cpu().numpy(), sigma2.cpu().numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests():
    print('Running GCMC unit tests...')
    torch.manual_seed(0)
    model = GCMC(tau_init=0.05)

    # ── Test 1: parameter count ───────────────────────────────────────────
    n_params = model.count_parameters()
    assert n_params == 213, f'Expected 213 params, got {n_params}'
    print(f'  [PASS] param count = {n_params}')

    # ── Test 2: permutation invariance ───────────────────────────────────
    states = torch.rand(6, 8)
    idx    = torch.randperm(6)

    dv1, s1, f1 = model(states)
    dv2, s2, f2 = model(states[idx])

    # f_i for track 0 should equal f_i for same track after shuffle
    # Compare per-track (match by index)
    inv_idx = torch.argsort(idx)
    assert torch.allclose(f1, f2[inv_idx], atol=1e-5), 'Aggregation not permutation-invariant'
    print('  [PASS] permutation invariance')

    # ── Test 3: k=0 (single track) → zero correction ─────────────────────
    single = torch.rand(1, 8)
    dv_s, sig_s, _ = model(single)
    assert torch.allclose(dv_s, torch.zeros(1, 2)), 'k=0: Δv should be zero'
    print('  [PASS] k=0 edge case (Δv=0)')

    # ── Test 4: k=1 (two tracks) → σ_i=0 in aggregation ─────────────────
    two    = torch.rand(2, 8)
    _, _, f_two = model(two)
    # σ dims (4:8) should be near zero (only one neighbour each)
    assert f_two[:, 4:].abs().max().item() < 1e-2, 'k=1: std should be ~0'
    print('  [PASS] k=1 edge case (σ≈0)')

    # ── Test 5: fanout energy preservation ───────────────────────────────
    dv_test = torch.tensor([[0.1, 0.2], [-0.3, 0.4]])
    fo = fanout(dv_test)
    # Sum of position corrections per track = Δv (two corners × 0.5 each)
    pos_sum_x = fo[:, 0] + fo[:, 2]
    pos_sum_y = fo[:, 1] + fo[:, 3]
    assert torch.allclose(pos_sum_x, dv_test[:, 0], atol=1e-6), 'fanout x energy'
    assert torch.allclose(pos_sum_y, dv_test[:, 1], atol=1e-6), 'fanout y energy'
    print('  [PASS] fanout energy conservation')

    # ── Test 6: sigma2 strictly positive ─────────────────────────────────
    states6 = torch.rand(10, 8)
    _, sig6, _ = model(states6)
    assert (sig6 > 0).all(), 'sigma2 must be strictly positive'
    print('  [PASS] sigma2 > 0')

    # ── Test 7: Q_aug shape and symmetry ─────────────────────────────────
    Q_base = np.eye(8) * 0.01
    sigma2_np = np.random.rand(5, 2) * 0.1
    Q_aug = build_Q_aug(Q_base, sigma2_np)
    assert Q_aug.shape == (5, 8, 8), f'Q_aug shape wrong: {Q_aug.shape}'
    # Diagonal should be >= Q_base diagonal
    for i in range(5):
        assert (np.diag(Q_aug[i]) >= np.diag(Q_base) - 1e-9).all()
    print('  [PASS] Q_aug shape and monotonicity')

    # ── Test 8: NLL loss decreases with correct prediction ───────────────
    dv_p  = torch.tensor([[0.1, 0.2]])
    dv_gt = torch.tensor([[0.1, 0.2]])   # perfect prediction
    sig   = torch.tensor([[0.01, 0.01]])
    loss_perfect = nll_loss(dv_p, dv_gt, sig).item()

    dv_wrong = torch.tensor([[0.5, 0.5]])
    loss_wrong = nll_loss(dv_wrong, dv_gt, sig).item()
    assert loss_perfect < loss_wrong, 'NLL should be lower for correct prediction'
    print('  [PASS] NLL loss monotonicity')

    print(f'\nAll tests passed. Model parameters: {n_params}')
    print(f'  τ_init = {model.aggregation.tau.item():.4f}')


if __name__ == '__main__':
    run_unit_tests()