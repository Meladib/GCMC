"""
train_gcmc.py — v10
==================
Changes from v9:
  - NeighbourAggregation removed — own-state 8-dim input only
  - tau logging removed (no aggregation module)
  - set_phase_a docstring updated
"""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gcmc import GCMC, nll_loss

DV_STD         = np.array([0.004789, 0.006376], dtype=np.float32)
DV_P50         = 0.003615
# V9: restored to V7 schedule
PHASE_A_EPOCHS = 40
PHASE_B_EPOCHS = 30   # total = 70


def set_phase_a(model):
    """Correction + backbone trainable. Uncertainty frozen."""
    for p in model.parameters():
        p.requires_grad = True
    for p in model.mlp.uncertainty_head.parameters():
        p.requires_grad = False


def set_phase_b(model):
    """V9 Change 2: ONLY uncertainty head trains. Backbone + correction frozen."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mlp.uncertainty_head.parameters():
        p.requires_grad = True


class ResidualDataset(Dataset):
    def __init__(self, npz_dir: Path, balanced=True, seed=42):
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
            for seq in per_seq: self.samples.extend(seq)
        else:
            rng = random.Random(seed)
            mx = max(len(s) for s in per_seq)
            for seq in per_seq:
                ov = seq * (mx // len(seq) + 1); rng.shuffle(ov)
                self.samples.extend(ov[:mx])

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_frames(batch): return batch


def compute_epoch_metrics(all_pred, all_gt, all_sigma2, all_errors_std, group_sizes):
    dv_pred = np.concatenate(all_pred,       axis=0)
    dv_gt   = np.concatenate(all_gt,         axis=0)
    sigma2  = np.concatenate(all_sigma2,     axis=0)
    err_std = np.concatenate(all_errors_std, axis=0)
    # Expand group sizes to per-track (each track gets its frame's group size)
    gs = np.repeat(group_sizes, group_sizes)

    errors    = dv_pred - dv_gt
    gt_norm   = np.linalg.norm(dv_gt,   axis=1)
    pred_norm = np.linalg.norm(dv_pred, axis=1)

    var_gt = np.var(dv_gt, axis=0).sum()
    mse    = (errors**2).mean()
    r2     = float(1.0 - mse / (var_gt + 1e-8))

    dot     = (dv_pred * dv_gt).sum(axis=1)
    norms   = pred_norm * gt_norm + 1e-8
    cos_sim = float((dot / norms).mean())

    mse_std          = (err_std**2).mean()
    sigma2_std_ratio = float(sigma2.mean() / max(mse_std, 1e-8))
    corr_ratio       = float(pred_norm.mean() / (gt_norm.mean() + 1e-8))
    activation_rate  = float((pred_norm > 0.001).mean())

    q75 = np.percentile(gt_norm, 75)
    q25 = np.percentile(gt_norm, 25)

    def safe_mse(mask): return float((errors[mask]**2).mean()) if mask.any() else 0.0

    return {
        'r2':                   r2,
        'cos_sim':              cos_sim,
        'corr_ratio':           corr_ratio,
        'activation_rate':      activation_rate,
        'mse':                  float(mse),
        'hard_mse':             safe_mse(gt_norm >= q75),
        'easy_mse':             safe_mse(gt_norm <= q25),
        'sharpness':            float(sigma2.mean()),
        'sigma2_mse_ratio_std': sigma2_std_ratio,
        'error_term':           float(((err_std**2) / sigma2).mean()),
        'uncertainty_term':     float(np.log(sigma2).mean()),
        'mse_small':            safe_mse(gs < 5),
        'mse_medium':           safe_mse((gs >= 5) & (gs < 15)),
        'mse_large':            safe_mse(gs >= 15),
    }


DV_STD_T = None


def run_epoch(model, loader, optimizer, device, epoch, phase):
    global DV_STD_T
    if DV_STD_T is None:
        DV_STD_T = torch.tensor(DV_STD, device=device)

    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    all_pred, all_gt, all_sigma2, all_errors_std, group_sizes = [], [], [], [], []

    for batch in loader:
        if training:
            optimizer.zero_grad()

        batch_pred, batch_gt, batch_sigma2, batch_err_std = [], [], [], []

        for states_np, dv_np in batch:
            s_t  = torch.tensor(states_np, dtype=torch.float32, device=device)
            dv_t = torch.tensor(dv_np,     dtype=torch.float32, device=device)

            dv_pred, sigma2, _ = model(s_t)

            if phase == 'A':
                # MSE in original space
                loss = ((dv_pred - dv_t)**2).mean()
            else:
                # V9 Change 2: dv_pred detached — σ² head sees fixed target
                dv_std = (dv_pred.detach() / DV_STD_T)
                gt_std = (dv_t / DV_STD_T)
                loss   = nll_loss(dv_std, gt_std, sigma2)

            total_loss += loss.item()

            dv_pred_np = dv_pred.detach().cpu().numpy()
            dv_gt_np   = dv_t.cpu().numpy()
            sigma2_np  = sigma2.detach().cpu().numpy()
            err_std_np = ((dv_pred.detach() - dv_t) / DV_STD_T).cpu().numpy()

            batch_pred.append(dv_pred_np)
            batch_gt.append(dv_gt_np)
            batch_sigma2.append(sigma2_np)
            batch_err_std.append(err_std_np)
            group_sizes.append(len(states_np))

            if training:
                loss.backward()

        if training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        all_pred.extend(batch_pred)
        all_gt.extend(batch_gt)
        all_sigma2.extend(batch_sigma2)
        all_errors_std.extend(batch_err_std)
        n_batches += 1

    metrics = compute_epoch_metrics(all_pred, all_gt, all_sigma2, all_errors_std, group_sizes)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics


def gate_check(metrics, phase, epoch):
    checks = {}
    if phase == 'A':
        checks['Phase A final R² > 0.10']         = (metrics['r2'] > 0.10,         metrics['r2'])
        checks['Phase A cos_sim > 0.20']           = (metrics['cos_sim'] > 0.20,    metrics['cos_sim'])
        checks['Phase A activation_rate > 0.70']   = (metrics['activation_rate'] > 0.70,
                                                       metrics['activation_rate'])
    else:
        checks['Phase B σ²/MSE_std 0.5-5x']       = (0.5 < metrics['sigma2_mse_ratio_std'] < 5.0,
                                                       metrics['sigma2_mse_ratio_std'])
        checks['Phase B R² ≥ Phase A R²']          = (True, metrics['r2'])  # always log

    all_pass = True
    for name, (passed, val) in checks.items():
        status = '[PASS]' if passed else '[FAIL]'
        print(f'  {status} {name}: {val:.4f}')
        if not passed:
            all_pass = False
    return all_pass


def train(args):
    device = args.device
    res_dir = Path(args.residuals)

    print(f'Loading residuals from {res_dir}')
    train_ds = ResidualDataset(res_dir / 'train')
    val_ds   = ResidualDataset(res_dir / 'val', balanced=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, collate_fn=collate_frames)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=collate_frames)

    model = GCMC().to(device)
    print(f'Model params: {model.count_parameters()}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    best_val_r2  = -999.0
    phase_a_r2   = None

    total_epochs = PHASE_A_EPOCHS + PHASE_B_EPOCHS

    for epoch in range(1, total_epochs + 1):
        phase = 'A' if epoch <= PHASE_A_EPOCHS else 'B'

        if epoch == 1:
            set_phase_a(model)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr)
            print(f'\n--- Phase A (epochs 1-{PHASE_A_EPOCHS}) ---')

        if epoch == PHASE_A_EPOCHS + 1:
            # Gate check
            print(f'\n=== Phase A Gate Check (epoch {PHASE_A_EPOCHS}) ===')
            gate_check(train_metrics, 'A', epoch)
            phase_a_r2 = train_metrics['r2']

            # V9 Change 2: Phase B — only uncertainty_head trains
            set_phase_b(model)
            optimizer = torch.optim.Adam(
                model.mlp.uncertainty_head.parameters(), lr=args.lr)
            print(f'\n--- Phase B (epochs {PHASE_A_EPOCHS+1}-{total_epochs}) ---')
            print('    Backbone FROZEN. Only uncertainty_head trains on detached NLL.')

        train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, phase)
        val_metrics   = run_epoch(model, val_loader,   None,      device, epoch, phase)

        entry = {'epoch': epoch, 'phase': phase,
                 **{f'train_{k}': v for k, v in train_metrics.items()},
                 **{f'val_{k}':   v for k, v in val_metrics.items()}}
        log.append(entry)

        print(f'ep{epoch:03d} [{phase}] '
              f'r2={val_metrics["r2"]:.4f}  '
              f'corr_ratio={val_metrics["corr_ratio"]:.4f}  '
              f'cos_sim={val_metrics["cos_sim"]:.4f}  '
              f'σ²/MSE={val_metrics["sigma2_mse_ratio_std"]:.3f}  '
              f'act={val_metrics["activation_rate"]:.3f}')

        # Save best
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            torch.save({'epoch': epoch, 'phase': phase,
                        'state_dict': model.state_dict(),
                        'val_r2': best_val_r2},
                       out_dir / 'gcmc_best.pt')

        with open(out_dir / 'training_log.json', 'w') as f:
            json.dump(log, f, indent=2)

    # Final gate check
    print(f'\n=== Final Gate Check ===')
    passed = gate_check(val_metrics, 'B', total_epochs)
    if phase_a_r2 is not None:
        status = '[PASS]' if val_metrics['r2'] >= phase_a_r2 else '[FAIL]'
        print(f'  {status} Phase B R² ≥ Phase A R²: {val_metrics["r2"]:.4f} ≥ {phase_a_r2:.4f}')

    print(f'\nBest val R²: {best_val_r2:.4f}')
    print(f'Checkpoint: {out_dir}/gcmc_best.pt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--residuals', required=True)
    p.add_argument('--out',       required=True)
    p.add_argument('--epochs',    type=int,   default=PHASE_A_EPOCHS + PHASE_B_EPOCHS)
    p.add_argument('--batch',     type=int,   default=256)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--device',    default='cuda')
    train(p.parse_args())


if __name__ == '__main__':
    main()
