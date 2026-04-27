"""
train_gcmc.py — v7
==================
Key change from v6:
  Phase A (epochs 1–PHASE_A): correction head + backbone + τ trained, uncertainty FROZEN, MSE loss
  Phase B (epochs PHASE_A+1–end): correction FROZEN, uncertainty head trained, NLL with DETACHED dv_pred

New logs:
  - Per-head gradient norms every epoch
  - R² (direction-aware, true learning signal)
  - σ²/MSE ratio in correct space (standardized)
  - Directional accuracy (cosine similarity between pred and GT vectors)
  - Per-phase loss tracking
"""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gcmc import GCMC, nll_loss

DV_STD          = np.array([0.004789, 0.006376], dtype=np.float32)
DV_P50          = 0.003615
PHASE_A_EPOCHS  = 60
PHASE_B_EPOCHS  = 40   # total = 100


def set_phase_a(model):
    """Correction + backbone + τ trainable. Uncertainty frozen."""
    for p in model.parameters():
        p.requires_grad = True
    for p in model.mlp.uncertainty_head.parameters():
        p.requires_grad = False


def set_phase_b(model):
    """Everything frozen except uncertainty head."""
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
    dv_pred = np.concatenate(all_pred,   axis=0)   # original space
    dv_gt   = np.concatenate(all_gt,     axis=0)
    sigma2  = np.concatenate(all_sigma2, axis=0)   # standardized space
    err_std = np.concatenate(all_errors_std, axis=0)  # standardized space errors

    errors    = dv_pred - dv_gt
    gt_norm   = np.linalg.norm(dv_gt,   axis=1)
    pred_norm = np.linalg.norm(dv_pred, axis=1)
    err_norm  = np.linalg.norm(errors,  axis=1)
    gs        = np.array(group_sizes)

    # R² — direction-aware (true learning signal)
    var_gt = np.var(dv_gt, axis=0).sum()
    mse    = (errors**2).mean()
    r2     = float(1.0 - mse / (var_gt + 1e-8))

    # Directional accuracy: cosine similarity between pred and GT vectors
    dot     = (dv_pred * dv_gt).sum(axis=1)
    norms   = pred_norm * gt_norm + 1e-8
    cos_sim = float((dot / norms).mean())

    # σ²/MSE in standardized space (correct comparison)
    mse_std    = (err_std**2).mean()
    sigma2_std_ratio = float(sigma2.mean() / max(mse_std, 1e-8))

    corr_ratio      = float(pred_norm.mean() / (gt_norm.mean() + 1e-8))
    activation_rate = float((pred_norm > 0.001).mean())
    q75 = np.percentile(gt_norm, 75)
    q25 = np.percentile(gt_norm, 25)

    def safe_mse(mask): return float((errors[mask]**2).mean()) if mask.any() else 0.0

    return {
        'r2':                  r2,
        'cos_sim':             cos_sim,
        'corr_ratio':          corr_ratio,
        'activation_rate':     activation_rate,
        'mse':                 float(mse),
        'hard_mse':            safe_mse(gt_norm >= q75),
        'easy_mse':            safe_mse(gt_norm <= q25),
        'sharpness':           float(sigma2.mean()),
        'sigma2_mse_ratio_std': sigma2_std_ratio,
        'error_term':          float(((err_std**2) / sigma2).mean()),
        'uncertainty_term':    float(np.log(sigma2).mean()),
        'mse_small':           safe_mse(gs < 5),
        'mse_medium':          safe_mse((gs >= 5) & (gs < 15)),
        'mse_large':           safe_mse(gs >= 15),
    }


DV_STD_T = None

def run_epoch(model, loader, optimizer, device, epoch, collect_grads=True):
    global DV_STD_T
    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    all_pred, all_gt, all_sigma2, all_errors_std, group_sizes = [], [], [], [], []
    grad_stats_acc = {'grad_correction': 0., 'grad_uncertainty': 0.,
                      'grad_backbone': 0., 'grad_tau': 0.,
                      'grad_corr_unc_ratio': 0.}
    grad_steps = 0

    phase_a = epoch <= PHASE_A_EPOCHS
    dv_std_t = DV_STD_T

    for batch in loader:
        losses = []
        for (states_np, dv_np) in batch:
            states  = torch.from_numpy(states_np).to(device)
            dv_raw  = torch.from_numpy(dv_np).to(device)

            # filter near-zero
            mag  = dv_raw.norm(dim=1)
            keep = mag > DV_P50
            if keep.sum() < 2: continue
            states = states[keep]; dv_raw = dv_raw[keep]

            # standardize
            dv_gt_std = torch.clamp(dv_raw / dv_std_t, -5.0, 5.0)
            weights   = (dv_raw.norm(dim=1) / dv_std_t.norm()).clamp(0.1, 10.0)

            dv_pred, sigma2, _ = model(states)

            if phase_a:
                # Phase A: weighted MSE, uncertainty head frozen
                frame_loss = (weights * ((dv_pred - dv_gt_std)**2).mean(dim=1)).mean()
            else:
                # Phase B: NLL with DETACHED dv_pred → only σ² learns
                sq_err     = (dv_pred.detach() - dv_gt_std) ** 2
                frame_loss = (sq_err / sigma2 + torch.log(sigma2)).mean()

            losses.append(frame_loss)

            with torch.no_grad():
                dv_pred_orig = dv_pred * dv_std_t
                err_std_np   = (dv_pred - dv_gt_std).cpu().numpy()
                all_pred.append(dv_pred_orig.cpu().numpy())
                all_gt.append(dv_raw.cpu().numpy())
                all_sigma2.append(sigma2.cpu().numpy())
                all_errors_std.append(err_std_np)
                group_sizes.extend([states.shape[0]] * states.shape[0])

        if not losses: continue

        mean_loss = torch.stack(losses).mean()
        total_loss += mean_loss.item(); n_batches += 1

        if training:
            optimizer.zero_grad()
            mean_loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)

            if collect_grads and training:
                gs = model.get_gradient_stats()
                for k in grad_stats_acc: grad_stats_acc[k] += gs.get(k, 0.)
                grad_steps += 1

            optimizer.step()

    mean_loss = total_loss / max(n_batches, 1)
    extra     = compute_epoch_metrics(all_pred, all_gt, all_sigma2,
                                      all_errors_std, group_sizes) if all_pred else {}

    # Average gradient stats
    if grad_steps > 0:
        for k in grad_stats_acc: grad_stats_acc[k] /= grad_steps
        extra['grad_correction']       = grad_stats_acc['grad_correction']
        extra['grad_uncertainty']      = grad_stats_acc['grad_uncertainty']
        extra['grad_backbone']         = grad_stats_acc['grad_backbone']
        extra['grad_tau']              = grad_stats_acc['grad_tau']
        extra['grad_corr_unc_ratio']   = grad_stats_acc['grad_corr_unc_ratio']

    return mean_loss, extra.get('mse', 0.), extra


def main():
    global DV_STD_T
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', default='residuals')
    parser.add_argument('--out',       default='checkpoints_v7')
    parser.add_argument('--epochs',    type=int,   default=70)
    parser.add_argument('--batch',     type=int,   default=256)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--device',    default='cuda')
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    DV_STD_T = torch.from_numpy(DV_STD).to(args.device)

    train_ds = ResidualDataset(Path(args.residuals)/'train', balanced=True,  seed=args.seed)
    val_ds   = ResidualDataset(Path(args.residuals)/'val',   balanced=False, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_frames)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_frames)

    model = GCMC().to(args.device)
    set_phase_a(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7)

    total_epochs = PHASE_A_EPOCHS + PHASE_B_EPOCHS
    print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}  Params: {model.count_parameters()}')
    print(f'Phase A (MSE, correction): epochs 1–{PHASE_A_EPOCHS}')
    print(f'Phase B (NLL, σ² calib):   epochs {PHASE_A_EPOCHS+1}–{total_epochs}')
    print(f'\n{"Ep":>4} {"Ph":>3} {"Loss":>8} {"R²":>7} {"CosSim":>8} '
          f'{"MSE":>9} {"σ²/MSE":>7} {"ActR":>6} {"∇C/U":>7} {"τ":>7}')
    print('─' * 80)

    log, best_val, best_ep = [], float('inf'), 0

    for epoch in range(1, total_epochs + 1):

        # Phase transition
        if epoch == PHASE_A_EPOCHS + 1:
            set_phase_b(model)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)
            print(f'\n  ── Phase B start: correction FROZEN, σ² calibration begins ──\n')

        phase = 'A' if epoch <= PHASE_A_EPOCHS else 'B'
        tr_loss, tr_mse, tr_ex = run_epoch(model, train_ld, optimizer, args.device, epoch)
        val_loss, val_mse, vl_ex = run_epoch(model, val_ld, None, args.device, epoch, False)
        tau = model.aggregation.tau.item()
        scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val, best_ep = val_loss, epoch
            torch.save({'epoch': epoch, 'phase': phase,
                        'state_dict': model.state_dict(), 'val_loss': val_loss, 'tau': tau},
                       out_dir / 'gcmc_best.pt')

        entry = {'epoch': epoch, 'phase': phase,
                 'train_loss': tr_loss, 'val_loss': val_loss,
                 'train_mse': tr_mse,   'val_mse': val_mse,
                 **{f'tr_{k}': v for k, v in tr_ex.items()},
                 **{f'val_{k}': v for k, v in vl_ex.items()}}
        log.append(entry)

        r2   = tr_ex.get('r2', 0)
        cs   = tr_ex.get('cos_sim', 0)
        smr  = tr_ex.get('sigma2_mse_ratio_std', 0)
        ar   = tr_ex.get('activation_rate', 0)
        gcr  = tr_ex.get('grad_corr_unc_ratio', 0)
        mk   = ' ★' if is_best else ''
        print(f'{epoch:>4} {phase:>3} {tr_loss:>8.4f} {r2:>7.4f} {cs:>8.4f} '
              f'{tr_mse:>9.6f} {smr:>7.2f} {ar:>6.4f} {gcr:>7.4f} {tau:>7.5f}{mk}')

    torch.save({'epoch': total_epochs, 'state_dict': model.state_dict(),
                'val_loss': val_loss, 'tau': tau},
               out_dir / 'gcmc_final.pt')
    (out_dir / 'training_log.json').write_text(json.dumps(log, indent=2))
    shutil.copy(Path(args.residuals) / 'norm.json', out_dir / 'norm.json')

    print(f'\nBest: {best_val:.5f} at epoch {best_ep}  |  Final τ: {tau:.5f}')

    # Gate checks
    print('\nV7 Gate checks:')
    final_a = next((e for e in reversed(log) if e['phase']=='A'), log[-1])
    final_b = log[-1]
    r2_a  = final_a.get('tr_r2', 0)
    r2_b  = final_b.get('tr_r2', 0)
    cs_a  = final_a.get('tr_cos_sim', 0)
    smr_b = final_b.get('tr_sigma2_mse_ratio_std', 0)
    ar_a  = final_a.get('tr_activation_rate', 0)
    print(f'  [{"PASS" if r2_a > 0.10 else "FAIL"}] Phase A final R² > 0.10: {r2_a:.4f}')
    print(f'  [{"PASS" if cs_a > 0.20 else "FAIL"}] Phase A cos_sim > 0.20: {cs_a:.4f}')
    print(f'  [{"PASS" if ar_a > 0.70 else "FAIL"}] Phase A activation_rate > 0.70: {ar_a:.4f}')
    print(f'  [{"PASS" if 0.5 < smr_b < 5.0 else "FAIL"}] Phase B σ²/MSE_std 0.5-5x: {smr_b:.3f}')
    print(f'  [{"PASS" if r2_b >= r2_a else "FAIL"}] Phase B R² ≥ Phase A R²: {r2_b:.4f} ≥ {r2_a:.4f}')


if __name__ == '__main__':
    main()