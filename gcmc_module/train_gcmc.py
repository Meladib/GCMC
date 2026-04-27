"""
train_gcmc_upgraded.py — v3
===========================
Fixes:
- Near-zero sample filtering (below p50 of |Δv*|)
- Two-phase: MSE warmup epochs 1-15, NLL from epoch 16
- Sample weighting by |Δv*| magnitude
- Feature normalization handled inside gcmc.py

Usage:
    python train_gcmc_upgraded.py \
        --residuals residuals --out checkpoints_v3 \
        --epochs 50 --batch 256 --lr 1e-4 --device cuda
"""

from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gcmc import GCMC, nll_loss

# From norm.json / diag_data.txt
DV_STD   = np.array([0.004789, 0.006376], dtype=np.float32)
DV_P50   = 0.003615   # median |Δv*| — filter threshold
MSE_WARMUP_EPOCHS = 25


# ── Uncertainty head freeze helper ────────────────────────────────────────────
def set_uncertainty_head(model, requires_grad: bool):
    for p in model.mlp.uncertainty_head.parameters():
        p.requires_grad = requires_grad


# ── Dataset ───────────────────────────────────────────────────────────────────
class ResidualDataset(Dataset):
    def __init__(self, npz_dir: Path, balanced: bool = True, seed: int = 42):
        self.samples = []
        per_seq = []
        for npz in sorted(npz_dir.glob('*.npz')):
            d = np.load(npz, allow_pickle=True)
            seq = []
            for s, dv in zip(d['states_norm'], d['dv_star']):
                s  = np.array(s,  dtype=np.float32)
                dv = np.array(dv, dtype=np.float32)
                if len(s) >= 2:
                    seq.append((s, dv))
            if seq:
                per_seq.append(seq)

        if not balanced:
            for seq in per_seq:
                self.samples.extend(seq)
        else:
            rng = random.Random(seed)
            max_len = max(len(s) for s in per_seq)
            for seq in per_seq:
                ov = seq * (max_len // len(seq) + 1)
                rng.shuffle(ov)
                self.samples.extend(ov[:max_len])

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_frames(batch): return batch


# ── Epoch metrics ─────────────────────────────────────────────────────────────

def compute_epoch_metrics(all_pred, all_gt, all_sigma2, group_sizes):
    dv_pred = np.concatenate(all_pred,   axis=0)
    dv_gt   = np.concatenate(all_gt,     axis=0)
    sigma2  = np.concatenate(all_sigma2, axis=0)

    errors    = dv_pred - dv_gt
    err_norm  = np.linalg.norm(errors,  axis=1)
    gt_norm   = np.linalg.norm(dv_gt,   axis=1)
    pred_norm = np.linalg.norm(dv_pred, axis=1)

    corr_ratio     = pred_norm.mean() / (gt_norm.mean() + 1e-8)
    activation_rate = (pred_norm > 0.001).mean()
    q75 = np.percentile(gt_norm, 75)
    q25 = np.percentile(gt_norm, 25)

    def safe_mse(mask):
        return float((errors[mask]**2).mean()) if mask.any() else 0.0

    gs = np.array(group_sizes)
    return {
        'corr_ratio':      float(corr_ratio),
        'activation_rate': float(activation_rate),
        'hard_mse':        safe_mse(gt_norm >= q75),
        'easy_mse':        safe_mse(gt_norm <= q25),
        'error_term':      float(((errors**2) / sigma2).mean()),
        'uncertainty_term':float(np.log(sigma2).mean()),
        'sharpness':       float(sigma2.mean()),
        'mse_small':       safe_mse(gs < 5),
        'mse_medium':      safe_mse((gs >= 5) & (gs < 15)),
        'mse_large':       safe_mse(gs >= 15),
    }


# ── Train / eval ──────────────────────────────────────────────────────────────

DV_STD_T = None   # set in main after device is known

def run_epoch(model, loader, optimizer, device, epoch=1):
    global DV_STD_T
    training = optimizer is not None
    model.train(training)

    total_nll, total_mse, n_batches = 0.0, 0.0, 0
    all_pred, all_gt, all_sigma2, group_sizes = [], [], [], []

    dv_std_t = DV_STD_T  # (2,) tensor on device

    for batch in loader:
        losses = []

        for (states_np, dv_np) in batch:
            states  = torch.from_numpy(states_np).to(device)   # (N,8)
            dv_raw  = torch.from_numpy(dv_np).to(device)       # (N,2)

            # ── Fix 3: filter near-zero samples ──────────────────────────
            mag  = dv_raw.norm(dim=1)
            keep = mag > DV_P50
            if keep.sum() < 2:
                continue
            states = states[keep]
            dv_raw = dv_raw[keep]

            # ── Fix 1: standardize targets ────────────────────────────────
            dv_gt = torch.clamp(dv_raw / dv_std_t, -5.0, 5.0)

            dv_pred, sigma2, _ = model(states)

            # ── Fix 2: sample weighting by |Δv*| ─────────────────────────
            weights = (dv_raw.norm(dim=1) / dv_std_t.norm()).clamp(0.1, 10.0)

            if epoch <= MSE_WARMUP_EPOCHS:
                frame_loss = (weights * ((dv_pred - dv_gt)**2).mean(dim=1)).mean()
            else:
                frame_loss = (weights * (((dv_pred - dv_gt)**2 / sigma2)
                                         + torch.log(sigma2)).mean(dim=1)).mean()

            losses.append(frame_loss)

            with torch.no_grad():
                # Rescale pred back to original space for metrics
                dv_pred_orig = dv_pred * dv_std_t
                dv_gt_orig   = dv_raw
                all_pred.append(dv_pred_orig.cpu().numpy())
                all_gt.append(dv_gt_orig.cpu().numpy())
                all_sigma2.append(sigma2.cpu().numpy())
                group_sizes.extend([states.shape[0]] * states.shape[0])
                total_mse += ((dv_pred_orig - dv_gt_orig)**2).mean().item()

        if not losses:
            continue

        mean_loss = torch.stack(losses).mean()
        total_nll += mean_loss.item()
        n_batches += 1

        if training:
            optimizer.zero_grad()
            mean_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    mean_nll = total_nll / max(n_batches, 1)
    mean_mse = total_mse / max(n_batches, 1)
    extra = compute_epoch_metrics(all_pred, all_gt, all_sigma2, group_sizes) \
            if all_pred else {}
    return mean_nll, mean_mse, extra


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DV_STD_T
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', default='residuals')
    parser.add_argument('--out',       default='checkpoints_v3')
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--batch',     type=int,   default=256)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--device',    default='cuda')
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    DV_STD_T = torch.from_numpy(DV_STD).to(args.device)

    train_ds = ResidualDataset(Path(args.residuals)/'train', balanced=True,  seed=args.seed)
    val_ds   = ResidualDataset(Path(args.residuals)/'val',   balanced=False, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_frames)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_frames)

    model     = GCMC().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    print(f'Train frames: {len(train_ds):,}  |  Val: {len(val_ds):,}')
    print(f'Params: {model.count_parameters()}  |  Device: {args.device}')
    print(f'MSE warmup: epochs 1-{MSE_WARMUP_EPOCHS}  →  NLL: epochs {MSE_WARMUP_EPOCHS+1}-{args.epochs}')
    print(f'Filter threshold: |Δv*| > {DV_P50:.6f} (p50)')
    print(f'\n{"Ep":>4} {"Phase":>5} {"TrainL":>9} {"ValL":>9} '
          f'{"MSE":>9} {"Sharp":>9} {"ActRate":>8} {"CorrR":>7} {"τ":>7}')
    print('─' * 75)

    log, best_val, best_ep = [], float('inf'), 0

    for epoch in range(1, args.epochs + 1):
        # Freeze/unfreeze uncertainty head at phase boundaries
        if epoch == 1:
            set_uncertainty_head(model, False)
            print('  [Uncertainty head FROZEN for MSE warmup]')
        elif epoch == MSE_WARMUP_EPOCHS + 1:
            set_uncertainty_head(model, True)
            print('  [Uncertainty head UNFROZEN for NLL phase]')
        
        phase = 'MSE' if epoch <= MSE_WARMUP_EPOCHS else 'NLL'
        tr_nll, tr_mse, tr_ex  = run_epoch(model, train_ld, optimizer, args.device, epoch)
        val_nll, val_mse, vl_ex = run_epoch(model, val_ld,  None,      args.device, epoch)
        tau = model.aggregation.tau.item()
        scheduler.step(val_nll)

        is_best = val_nll < best_val
        if is_best:
            best_val, best_ep = val_nll, epoch
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_nll, 'tau': tau},
                       out_dir / 'gcmc_best.pt')

        entry = {'epoch': epoch, 'phase': phase,
                 'train_nll': tr_nll, 'val_nll': val_nll,
                 'train_mse': tr_mse, 'val_mse': val_mse,
                 **{f'tr_{k}': v for k, v in tr_ex.items()},
                 **{f'val_{k}': v for k, v in vl_ex.items()}}
        log.append(entry)

        ar = tr_ex.get('activation_rate', 0)
        cr = tr_ex.get('corr_ratio', 0)
        sh = tr_ex.get('sharpness', 0)
        mk = ' ★' if is_best else ''
        print(f'{epoch:>4} {phase:>5} {tr_nll:>9.4f} {val_nll:>9.4f} '
              f'{tr_mse:>9.6f} {sh:>9.6f} {ar:>8.4f} {cr:>7.4f} {tau:>7.5f}{mk}')

    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(),
                'val_loss': val_nll, 'tau': tau},
               out_dir / 'gcmc_final.pt')
    (out_dir / 'training_log.json').write_text(json.dumps(log, indent=2))
    shutil.copy(Path(args.residuals) / 'norm.json', out_dir / 'norm.json')

    print(f'\nBest val loss: {best_val:.5f} at epoch {best_ep}')
    print(f'Final τ: {tau:.5f}')

    # Gate checks
    print('\nPhase 5 gate checks:')
    if len(log) > MSE_WARMUP_EPOCHS:
        e15 = log[MSE_WARMUP_EPOCHS-1]
        e16 = log[MSE_WARMUP_EPOCHS]
        ok = e15.get('tr_corr_ratio', 0) > e15.get('tr_corr_ratio', 0) or \
             e15.get('tr_activation_rate', 0) > 0
        print(f'  [{"PASS" if ok else "CHECK"}] Activation rate after MSE warmup: '
              f'{e15.get("tr_activation_rate",0):.4f}')
    final = log[-1]
    cr  = final.get('tr_corr_ratio', 0)
    ar  = final.get('tr_activation_rate', 0)
    sh  = final.get('tr_sharpness', 0)
    print(f'  [{"PASS" if ar > 0.01 else "FAIL"}] activation_rate > 0.01: {ar:.4f}')
    print(f'  [{"PASS" if cr > 0.3  else "FAIL"}] corr_ratio > 0.3: {cr:.4f}')
    print(f'  [{"PASS" if sh > 1e-4 else "FAIL"}] no σ² collapse (sharpness > 1e-4): {sh:.6f}')

if __name__ == '__main__':
    main()
