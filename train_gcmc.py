"""
train_gcmc.py — Phase 5: Train GCMC module
==========================================
Usage:
    python train_gcmc.py \
        --residuals residuals \
        --out checkpoints \
        --epochs 50 \
        --batch 256 \
        --lr 1e-3 \
        --device cuda

Outputs:
    checkpoints/
        gcmc_best.pt        ← best val NLL checkpoint
        gcmc_final.pt       ← last epoch checkpoint
        training_log.json   ← per-epoch metrics
        norm.json           ← copy of residuals/norm.json (for inference)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gcmc import GCMC, nll_loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ResidualDataset(Dataset):
    """Loads all .npz residual files and flattens to (states, dv_star) pairs.

    Sequence-balanced sampling: each sequence contributes equally regardless
    of length — avoids dancetrack0020 (40 dancers, 583 frames) dominating.
    """

    def __init__(self, npz_dir: Path, balanced: bool = True, seed: int = 42) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        per_seq: list[list[tuple[np.ndarray, np.ndarray]]] = []

        for npz_path in sorted(npz_dir.glob('*.npz')):
            data        = np.load(npz_path, allow_pickle=True)
            states_list = data['states_norm']   # object array of (N_t, 8)
            dv_list     = data['dv_star']        # object array of (N_t, 2)

            seq_samples = []
            for states, dv in zip(states_list, dv_list):
                # Each frame: states (N_t, 8), dv_star (N_t, 2)
                if len(states) < 2:
                    continue
                seq_samples.append((
                    states.astype(np.float32),
                    dv.astype(np.float32),
                ))
            if seq_samples:
                per_seq.append(seq_samples)

        if not balanced:
            for seq in per_seq:
                self.samples.extend(seq)
        else:
            # Balanced: sample same number of frames from each sequence
            rng = random.Random(seed)
            max_per_seq = max(len(s) for s in per_seq)
            for seq in per_seq:
                # Oversample short sequences to reach max_per_seq
                oversampled = seq * (max_per_seq // len(seq) + 1)
                rng.shuffle(oversampled)
                self.samples.extend(oversampled[:max_per_seq])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_frames(batch):
    """Each item is (states: (N,8), dv: (N,2)) with variable N.
    Returns list of tensors — handled per-frame in training loop.
    """
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metric (ECE proxy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(
    all_errors: list[np.ndarray],
    all_sigma2: list[np.ndarray],
) -> float:
    """Pearson correlation between |Δv error| and predicted σ.
    Positive correlation → σ² tracks actual uncertainty.
    """
    errors = np.concatenate(all_errors, axis=0).mean(axis=1)  # (M,)
    sigmas = np.concatenate(all_sigma2, axis=0).mean(axis=1)  # (M,)
    if len(errors) < 2:
        return 0.0
    corr = np.corrcoef(np.abs(errors), np.sqrt(sigmas))[0, 1]
    return float(corr)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval one epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     GCMC,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    str,
) -> tuple[float, float, float]:
    """Returns (mean_nll, mean_mse, calibration_corr)."""
    training = optimizer is not None
    model.train(training)

    total_nll, total_mse, n_samples = 0.0, 0.0, 0
    all_errors, all_sigma2 = [], []

    for batch in loader:
        batch_nll, batch_mse = 0.0, 0.0
        batch_n = 0

        for (states_np, dv_np) in batch:
            states = torch.from_numpy(states_np).to(device)  # (N, 8)
            dv_gt = torch.from_numpy(dv_np).to(device)      # (N, 2)
            dv_gt = torch.clamp(dv_gt, -0.1, 0.1)

            dv_pred, sigma2, _ = model(states)

            loss = nll_loss(dv_pred, dv_gt, sigma2)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                mse = ((dv_pred - dv_gt) ** 2).mean().item()
                batch_nll += loss.item()
                batch_mse += mse
                batch_n   += 1

                err = (dv_pred - dv_gt).cpu().numpy()
                sig = sigma2.cpu().numpy()
                all_errors.append(err)
                all_sigma2.append(sig)

        if batch_n > 0:
            total_nll  += batch_nll
            total_mse  += batch_mse
            n_samples  += batch_n

    mean_nll  = total_nll  / max(n_samples, 1)
    mean_mse  = total_mse  / max(n_samples, 1)
    calib     = compute_calibration(all_errors, all_sigma2)
    return mean_nll, mean_mse, calib


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', default='residuals')
    parser.add_argument('--out',       default='checkpoints')
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--batch',     type=int,   default=256)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--tau-init',  type=float, default=0.05)
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--no-balanced', action='store_true',
                        help='Disable sequence-balanced sampling')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_dir = Path(args.residuals)

    # ── Datasets ──────────────────────────────────────────────────────────
    balanced = not args.no_balanced
    train_ds = ResidualDataset(res_dir / 'train', balanced=balanced, seed=args.seed)
    val_ds   = ResidualDataset(res_dir / 'val',   balanced=False,    seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  collate_fn=collate_frames,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=collate_frames,
                              num_workers=0)

    print(f'Train frames: {len(train_ds):,}  |  Val frames: {len(val_ds):,}')
    print(f'Device: {args.device}  |  Balanced sampling: {balanced}')

    # ── Model + optimiser ─────────────────────────────────────────────────
    model     = GCMC(tau_init=args.tau_init).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print(f'Parameters: {model.count_parameters()}  |  τ_init={args.tau_init}')

    # ── Training loop ─────────────────────────────────────────────────────
    log       = []
    best_nll  = float('inf')
    best_epoch = 0

    print(f'\n{"Ep":>4}  {"Train NLL":>10}  {"Val NLL":>10}  '
          f'{"Train MSE":>10}  {"Val MSE":>10}  '
          f'{"Calib":>7}  {"τ":>7}')
    print('─' * 70)

    for epoch in range(1, args.epochs + 1):

        tr_nll, tr_mse, _        = run_epoch(model, train_loader, optimizer, args.device)
        val_nll, val_mse, calib  = run_epoch(model, val_loader,   None,      args.device)

        tau_val = model.aggregation.tau.item()
        scheduler.step(val_nll)

        # ── Checkpoint ────────────────────────────────────────────────────
        is_best = val_nll < best_nll
        if is_best:
            best_nll   = val_nll
            best_epoch = epoch
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_nll':    val_nll,
                'tau':        tau_val,
            }, out_dir / 'gcmc_best.pt')

        entry = {
            'epoch':    epoch,
            'train_nll': round(tr_nll,  6),
            'val_nll':   round(val_nll, 6),
            'train_mse': round(tr_mse,  8),
            'val_mse':   round(val_mse, 8),
            'calib':     round(calib,   4),
            'tau':       round(tau_val, 6),
            'best':      is_best,
        }
        log.append(entry)

        marker = ' ★' if is_best else ''
        print(f'{epoch:>4}  {tr_nll:>10.5f}  {val_nll:>10.5f}  '
              f'{tr_mse:>10.7f}  {val_mse:>10.7f}  '
              f'{calib:>7.4f}  {tau_val:>7.5f}{marker}')

    # ── Final checkpoint ──────────────────────────────────────────────────
    torch.save({
        'epoch':      args.epochs,
        'state_dict': model.state_dict(),
        'val_nll':    val_nll,
        'tau':        model.aggregation.tau.item(),
    }, out_dir / 'gcmc_final.pt')

    # ── Save log + norm ───────────────────────────────────────────────────
    (out_dir / 'training_log.json').write_text(json.dumps(log, indent=2))
    shutil.copy(res_dir / 'norm.json', out_dir / 'norm.json')

    print(f'\n{"─"*70}')
    print(f'Best val NLL: {best_nll:.5f} at epoch {best_epoch}')
    print(f'Final τ:      {model.aggregation.tau.item():.5f}')
    print(f'Saved to:     {out_dir.resolve()}')

    # ── Phase 5 gate checks ───────────────────────────────────────────────
    print('\nPhase 5 gate checks:')

    # 1. Val NLL < train NLL by epoch 10
    if len(log) >= 10:
        e10 = log[9]
        ok1 = e10['val_nll'] < e10['train_nll']
        print(f'  [{"PASS" if ok1 else "WARN"}] Val NLL < Train NLL at epoch 10  '
              f'(val={e10["val_nll"]:.5f}, train={e10["train_nll"]:.5f})')

    # 2. τ converged (Δτ < 0.001 over last 10 epochs)
    if len(log) >= 10:
        tau_last10 = [e['tau'] for e in log[-10:]]
        delta_tau  = max(tau_last10) - min(tau_last10)
        ok2 = delta_tau < 0.001
        print(f'  [{"PASS" if ok2 else "WARN"}] τ converged  '
              f'(Δτ={delta_tau:.6f} over last 10 epochs)')

    # 3. Calibration > 0 (σ² positively correlated with error)
    final_calib = log[-1]['calib']
    ok3 = final_calib > 0
    print(f'  [{"PASS" if ok3 else "WARN"}] σ² calibration correlation > 0  '
          f'(corr={final_calib:.4f})')

    # 4. Best epoch not epoch 1 (model actually learned)
    ok4 = best_epoch > 1
    print(f'  [{"PASS" if ok4 else "WARN"}] Best epoch > 1  (best={best_epoch})')


if __name__ == '__main__':
    main()