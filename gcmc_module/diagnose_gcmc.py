"""
diagnose_gcmc.py — GCMC Diagnostic v9
======================================
V9 changes vs v3:
  - DIM_NAMES 8-9: pos_spread_x/y (was nb_gvx/gvy)
  - _build_features: pos_spread replaces group_vel, vel_dev replaces own_vel in states_rel
  - _normalize: dims 8-9 use POS_SCALE (was VEL_SCALE)
  - §3: CorrR reads from 'corr_ratio' key (fixes zero-read bug)
  - §5D: confidence gate sweep (replaces pixel threshold sweep)
  - §6: failure taxonomy uses confidence gate
  - §7: recommendations updated — implemented items removed

Usage:
    python diagnose_gcmc.py --residuals residuals --log checkpoints_v9/training_log.json --mode full
    python diagnose_gcmc.py --residuals residuals --log checkpoints_v9/training_log.json \
        --checkpoint checkpoints_v9/gcmc_best.pt --mode report
    python diagnose_gcmc.py --residuals residuals --mode compare \
        --log checkpoints_v7/training_log.json --log2 checkpoints_v9/training_log.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

SEP  = '─' * 70
SEP2 = '═' * 70

POS_SCALE        = 0.2257
VEL_SCALE        = 0.0022
DV_STD           = np.array([0.004789, 0.006376], dtype=np.float32)
DV_CONF_THRESHOLD = 1.0   # V9 confidence gate threshold

# V9 dim names — dims 8-9 now pos_spread, dims 16-19 now vel_deviation
DIM_NAMES = [
    'nb_μ_dx',     'nb_μ_dy',     'nb_μ_dvx',  'nb_μ_dvy',
    'nb_σ_dx',     'nb_σ_dy',     'nb_σ_dvx',  'nb_σ_dvy',
    'pos_spread_x','pos_spread_y','vel_dev_x',  'vel_dev_y',   # V9: was gvx,gvy,devx,devy
    'own_x1',      'own_y1',      'own_x2',     'own_y2',
    'vel_devx1',   'vel_devy1',   'vel_devx2',  'vel_devy2',   # V9: own_vel - group_vel
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trend(values, window=5):
    if len(values) < window: return 0.0
    x = np.arange(window); y = np.array(values[-window:])
    return float(np.polyfit(x, y, 1)[0])


def _load_all_residuals(res_dir, split='train', max_seqs=None):
    split_dir = Path(res_dir) / split
    if not split_dir.exists(): return []
    files = sorted(split_dir.glob('*.npz'))
    if max_seqs: files = files[:max_seqs]
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for s, dv in zip(d['states_norm'], d['dv_star']):
            s  = np.array(s,  dtype=np.float32)
            dv = np.array(dv, dtype=np.float32)
            if len(s) >= 2:
                out.append((s, dv, f.stem))
    return out


def _build_features(states):
    """V9 feature construction — matches gcmc.py V9 forward()."""
    N   = len(states)
    pos = (states[:, :2] + states[:, 2:4]) / 2.0
    vel = (states[:, 4:6] + states[:, 6:8]) / 2.0

    dpos = pos[:, None] - pos[None, :]
    dvel = vel[:, None] - vel[None, :]
    f_ij = np.concatenate([dpos, dvel], axis=-1)

    dist = np.sqrt((dpos**2).sum(-1) + 1e-8)
    np.fill_diagonal(dist, 1e9)
    tau  = 0.05
    logits = -dist / tau
    logits -= logits.max(axis=1, keepdims=True)
    alpha  = np.exp(logits); alpha /= alpha.sum(axis=1, keepdims=True)

    mu    = np.einsum('ij,ijd->id', alpha, f_ij)
    diff2 = (f_ij - mu[:, None])**2
    sigma = np.sqrt(np.einsum('ij,ijd->id', alpha, diff2) + 1e-6)

    group_vel  = vel.mean(axis=0, keepdims=True).repeat(N, axis=0)
    vel_dev    = vel - group_vel                            # (N,2) — dims 10-11
    # V9: pos_spread replaces group_vel (dims 8-9)
    pos_spread = vel.std(axis=0, keepdims=True).repeat(N, axis=0)  # broadcast same shape

    f_nb = np.concatenate([mu, sigma, pos_spread, vel_dev], axis=-1)  # (N,12)

    # V9: own vel as deviation from group
    states_rel = states.copy()
    states_rel[:, 4:6] = states[:, 4:6] - group_vel
    states_rel[:, 6:8] = states[:, 6:8] - group_vel

    f_i = np.concatenate([f_nb, states_rel], axis=-1)  # (N,20)
    return f_i, alpha


def _normalize_features(f_i):
    """V9 normalisation — dims 8-9 use POS_SCALE (was VEL_SCALE)."""
    f = f_i.copy()
    f[:, 0:2]  /= POS_SCALE
    f[:, 2:4]  /= VEL_SCALE
    f[:, 4:6]  /= POS_SCALE
    f[:, 6:8]  /= VEL_SCALE
    f[:, 8:10] /= POS_SCALE   # V9: pos_spread → POS_SCALE
    f[:, 10:12] /= VEL_SCALE
    f[:, 12:16] /= POS_SCALE
    f[:, 16:20] /= VEL_SCALE
    return f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Residual Data Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_data(res_dir):
    print(f'\n{SEP2}')
    print('  SECTION 1: RESIDUAL DATA ANALYSIS')
    print(SEP2)

    for split in ['train', 'val']:
        frames = _load_all_residuals(res_dir, split)
        if not frames: continue

        all_dv_mag, all_vel_mag, group_sizes = [], [], []
        n_cold = 0
        seq_counts = defaultdict(int)

        for s, dv, seq in frames:
            dv_mag  = np.linalg.norm(dv, axis=1)
            vel     = (s[:, 4:6] + s[:, 6:8]) / 2.0
            vel_mag = np.sqrt((vel**2).sum(-1))
            all_dv_mag.extend(dv_mag.tolist())
            all_vel_mag.extend(vel_mag.tolist())
            group_sizes.append(len(s))
            if vel_mag.max() < 1e-4: n_cold += 1
            seq_counts[seq] += 1

        dv_arr  = np.array(all_dv_mag)
        vel_arr = np.array(all_vel_mag)
        gs_arr  = np.array(group_sizes)

        noise_floor = np.percentile(dv_arr, 50)
        signal      = np.percentile(dv_arr, 95)
        snr         = signal / max(noise_floor, 1e-8)

        print(f'\n  [{split.upper()}]')
        print(f'  {SEP}')
        print(f'  Frames           : {len(frames):,}  |  Tracks: {len(dv_arr):,}')
        print(f'  Sequences        : {len(seq_counts)}')
        print(f'  Group size       : mean={gs_arr.mean():.1f}  med={np.median(gs_arr):.0f}  '
              f'p5={np.percentile(gs_arr,5):.0f}  p95={np.percentile(gs_arr,95):.0f}')
        print(f'  Cold frames (v=0): {n_cold} / {len(frames)} ({100*n_cold/len(frames):.1f}%)')
        print(f'\n  Target |Δv*| distribution:')
        for pct in [25, 50, 75, 90, 95, 99]:
            print(f'     p{pct:<3}: {np.percentile(dv_arr, pct):.6f}')
        print(f'  SNR (p95/p50)    : {snr:.2f}x')
        if snr < 5:
            print(f'  ⚠ LOW SNR — sparse signal, zero-init dangerous')

        print(f'\n  Speed-stratified signal (|Δv*| p95 per speed bin):')
        for lo, hi in [(0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, 1.0)]:
            mask = (vel_arr >= lo) & (vel_arr < hi)
            if mask.sum() > 10:
                sig_bin = np.percentile(dv_arr[mask], 95)
                nf_bin  = np.percentile(dv_arr[mask], 50)
                print(f'     vel [{lo:.3f},{hi:.3f}): n={mask.sum():5d}  '
                      f'p50={nf_bin:.5f}  p95={sig_bin:.5f}  SNR={sig_bin/max(nf_bin,1e-9):.1f}x')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Model Architecture Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_model(res_dir):
    print(f'\n{SEP2}')
    print('  SECTION 2: MODEL ARCHITECTURE ANALYSIS (V9 features)')
    print(SEP2)

    frames = _load_all_residuals(res_dir, 'train', max_seqs=3)
    if not frames:
        print('  No training data found.'); return

    warm_frame = cold_frame = None
    for s, dv, _ in frames:
        vel = (s[:, 4:6] + s[:, 6:8]) / 2.0
        vm  = np.sqrt((vel**2).sum(-1)).max()
        if warm_frame is None and vm > 1e-4 and len(s) >= 3:
            warm_frame = (s, dv)
        if cold_frame is None and vm < 1e-5:
            cold_frame = (s, dv)

    for label, frame in [('cold', cold_frame), ('warm', warm_frame)]:
        if frame is None: continue
        s, dv = frame
        f_i, alpha = _build_features(s)
        f_norm     = _normalize_features(f_i)

        print(f'\n  ── Feature stats [{label}]  N={len(s)}')
        print(f'     {"Dim":<14} {"raw mean":>10} {"norm mean":>10} {"norm std":>10}')
        print(f'     {"-"*46}')
        for i, name in enumerate(DIM_NAMES):
            raw_m  = np.abs(f_i[:, i]).mean()
            norm_m = np.abs(f_norm[:, i]).mean()
            norm_s = f_norm[:, i].std()
            flag   = ' ⚠ dead' if norm_m < 0.01 else ''
            print(f'     {name:<14} {raw_m:>10.5f} {norm_m:>10.4f} {norm_s:>10.4f}{flag}')

        ent = -(alpha * np.log(alpha + 1e-9)).sum(-1).mean()
        print(f'\n     Attention entropy (mean): {ent:.4f}')
        if ent < 0.3:
            print(f'     ⚠ LOW entropy — nearly one-hot, context collapsed')
        elif ent > 2.0:
            print(f'     ⚠ HIGH entropy — too diffuse, neighbourhood blurred')

    # Gradient race check (untrained)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from gcmc import GCMC, nll_loss as gcmc_nll

        model = GCMC(tau_init=0.05)
        s_t  = torch.tensor(warm_frame[0] if warm_frame else frames[0][0], dtype=torch.float32)
        dv_t = torch.tensor(warm_frame[1] if warm_frame else frames[0][1], dtype=torch.float32)
        dv_pred, sigma2, _ = model(s_t)
        gcmc_nll(dv_pred, dv_t, sigma2).backward()

        def gnorm(params):
            gs = [p.grad.norm().item() for p in params if p.grad is not None]
            return sum(gs) / max(len(gs), 1)

        g_corr = gnorm(model.mlp.correction_head.parameters())
        g_unc  = gnorm(model.mlp.uncertainty_head.parameters())
        g_bb   = gnorm(model.mlp.backbone.parameters())
        ratio  = g_corr / max(g_unc, 1e-12)

        print(f'\n  ── Gradient race (NLL, untrained):')
        print(f'     correction_head: {g_corr:.6f}')
        print(f'     uncertainty_head:{g_unc:.6f}')
        print(f'     backbone:        {g_bb:.6f}')
        print(f'     corr/unc ratio:  {ratio:.4f}  '
              f'{"✓ balanced" if 0.1 < ratio < 10 else "⚠ RACE"}')

        w0 = model.mlp.backbone[0].weight
        if w0.grad is not None and w0.shape[1] == 20:
            g_nb  = w0.grad[:, :12].norm().item()
            g_own = w0.grad[:, 12:].norm().item()
            print(f'     own-state grad:  {g_own:.6f}  (dims 12-19)')
            print(f'     neighbour grad:  {g_nb:.6f}  (dims 0-11)')
            print(f'     own/neigh ratio: {g_own/max(g_nb,1e-12):.3f}')
        print(f'     param count:     {model.count_parameters()}')
    except ImportError:
        print('\n  [skip] gcmc.py not importable')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Training Log Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_log(log_path, label='run'):
    print(f'\n{SEP2}')
    print(f'  SECTION 3: TRAINING LOG — {label}')
    print(SEP2)

    with open(log_path) as f:
        log = json.load(f)
    if not log:
        print('  Empty log.'); return

    phases = defaultdict(list)
    for e in log:
        phases[e.get('phase', 'A')].append(e)

    for phase, entries in phases.items():
        epochs = [e['epoch'] for e in entries]
        r2s    = [e.get('tr_r2',    e.get('val_r2',    e.get('r2',    0))) for e in entries]
        # V9 fix: read corr_ratio key (was corr_r in broken v3)
        corr_rs = [e.get('tr_corr_ratio', e.get('val_corr_ratio',
                   e.get('corr_ratio',    e.get('corr_r', 0)))) for e in entries]
        ratios  = [e.get('tr_sigma2_mse_ratio_std', e.get('sigma2_mse_ratio_std', 0))
                   for e in entries]
        act_rates = [e.get('tr_activation_rate', e.get('activation_rate'))
                     for e in entries]
        act_rates = [a for a in act_rates if a is not None]

        peak_corr  = max(corr_rs)
        peak_ep    = epochs[corr_rs.index(peak_corr)]
        final_corr = corr_rs[-1]
        final_r2   = r2s[-1]
        trend_corr = _trend(corr_rs)

        print(f'\n  Phase {phase}  (epochs {epochs[0]}–{epochs[-1]})')
        print(f'  {SEP}')
        print(f'  R²        : start={r2s[0]:.4f}  peak={max(r2s):.4f}  final={final_r2:.4f}')
        print(f'  CorrR     : start={corr_rs[0]:.4f}  peak={peak_corr:.4f} @ep{peak_ep}  '
              f'final={final_corr:.4f}  trend={trend_corr:+.5f}/ep')
        if ratios[0] > 0:
            print(f'  σ²/MSE    : start={ratios[0]:.3f}  final={ratios[-1]:.3f}')
        if act_rates:
            print(f'  Act.Rate  : mean={np.mean(act_rates):.3f}  final={act_rates[-1]:.3f}')

        decay = peak_corr - final_corr
        if phase == 'B' and decay > 0.005:
            print(f'  ⚠ Phase B eroded CorrR by {decay:.4f}')
        if final_corr < 0.35:
            print(f'  ⚠ CorrR {final_corr:.4f} below dataset SNR ceiling (~0.35)')
        if abs(trend_corr) < 1e-5 and len(entries) > 10:
            print(f'  ⚠ CorrR flat for {len(entries)} epochs — local optimum')

    if 'A' in phases and 'B' in phases:
        a_final = phases['A'][-1]
        b_final = phases['B'][-1]
        a_corr  = a_final.get('tr_corr_ratio', a_final.get('corr_ratio', 0))
        b_corr  = b_final.get('tr_corr_ratio', b_final.get('corr_ratio', 0))
        delta   = b_corr - a_corr
        print(f'\n  Phase A→B CorrR delta: {delta:+.4f}  '
              f'{"✓ B improved" if delta > 0 else "⚠ B degraded"}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Feature Attribution
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_attribution(res_dir, checkpoint=None):
    print(f'\n{SEP2}')
    print('  SECTION 4: FEATURE ATTRIBUTION (V9 features)')
    print(SEP2)

    frames = _load_all_residuals(res_dir, 'val', max_seqs=8)
    if not frames:
        frames = _load_all_residuals(res_dir, 'train', max_seqs=8)
    if not frames:
        print('  No data found.'); return

    all_f, all_dv_x, all_dv_y = [], [], []
    for s, dv, _ in frames:
        f_i, _ = _build_features(s)
        all_f.append(_normalize_features(f_i))
        all_dv_x.extend(dv[:, 0].tolist())
        all_dv_y.extend(dv[:, 1].tolist())

    F   = np.concatenate(all_f, axis=0)
    dvx = np.array(all_dv_x)
    dvy = np.array(all_dv_y)
    dvm = np.sqrt(dvx**2 + dvy**2)

    print(f'\n  ── A. Input-target correlations (Pearson, val split)')
    print(f'  {"Dim":<14} {"corr(dvx)":>10} {"corr(dvy)":>10} {"corr(|dv|)":>10}  note')
    print(f'  {"-"*62}')
    dim_scores = []
    for i, name in enumerate(DIM_NAMES):
        fi = F[:, i]
        cx = np.corrcoef(fi, dvx)[0, 1]
        cy = np.corrcoef(fi, dvy)[0, 1]
        cm = np.corrcoef(fi, dvm)[0, 1]
        score = max(abs(cx), abs(cy), abs(cm))
        dim_scores.append((score, i, name, cx, cy, cm))
        flag = ('✓ STRONG' if score > 0.20 else '✓ info' if score > 0.08 else '  weak')
        print(f'  {name:<14} {cx:>10.4f} {cy:>10.4f} {cm:>10.4f}  {flag}')

    dim_scores.sort(reverse=True)
    print(f'\n  Top-5 predictive dims: '
          + ', '.join(f'{n}({s:.3f})' for s, _, n, *_ in dim_scores[:5]))
    dead = [n for s, _, n, *_ in dim_scores if s < 0.03]
    if dead:
        print(f'  Dead dims (<0.03):    {", ".join(dead)}')

    # V9: check pos_spread vs own_pos redundancy (replaces old group_vel check)
    print(f'\n  ── B. V9 feature redundancy checks')
    for nb_i, own_i, label in [
        (8,  12, 'pos_spread_x↔own_x1'),
        (9,  13, 'pos_spread_y↔own_y1'),
        (10, 16, 'vel_dev_x↔vel_devx1'),
        (11, 17, 'vel_dev_y↔vel_devy1'),
    ]:
        r = np.corrcoef(F[:, nb_i], F[:, own_i])[0, 1]
        print(f'     {label}: r={r:.4f}{"  ⚠ redundant" if abs(r) > 0.8 else ""}')

    if checkpoint:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from gcmc import GCMC
            model = GCMC(); model.eval()
            ckpt  = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))

            print(f'\n  ── C. Per-dim gradient sensitivity (trained checkpoint)')
            sens = np.zeros(20); n_used = 0
            for s, dv, _ in frames[:4]:
                if len(s) < 2: continue
                s_t = torch.tensor(s, dtype=torch.float32)
                with torch.enable_grad():
                    dv_pred, _, _ = model(s_t)
                    loss = dv_pred.pow(2).sum()
                    if s_t.grad is not None: s_t.grad.zero_()
                    loss.backward()
                n_used += 1
            if n_used > 0:
                print(f'  [skip] direct input grad not available via model.forward — use backbone weight grad')
        except Exception as e:
            print(f'  [skip] {e}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Correction Quality Profile
# ══════════════════════════════════════════════════════════════════════════════

def analyze_correction_quality(res_dir, checkpoint=None):
    print(f'\n{SEP2}')
    print('  SECTION 5: CORRECTION QUALITY PROFILE')
    print(SEP2)

    if not checkpoint:
        print('  No checkpoint — skipping (pass --checkpoint).'); return

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from gcmc import GCMC
        model = GCMC(); model.eval()
        ckpt  = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
    except Exception as e:
        print(f'  Cannot load model: {e}'); return

    frames = _load_all_residuals(res_dir, 'val', max_seqs=25)
    if not frames:
        print('  No val data.'); return

    pred_mags, gt_mags, cos_sims, confidences = [], [], [], []
    r2_num = r2_den = 0.0
    seq_errors = defaultdict(list)

    with torch.no_grad():
        for s, dv_gt, seq in frames:
            if len(s) < 2: continue
            s_t = torch.tensor(s, dtype=torch.float32)
            dv_pred, sigma2, _ = model(s_t)
            dv_p = dv_pred.numpy() * DV_STD
            """change
            pred_mag = np.linalg.norm(dv_p, axis=1)
            sigma2_np = sigma2.numpy()
            conf = pred_mag / (np.sqrt(sigma2_np.mean(axis=1)) + 1e-8)
            """

            pred_mag = np.linalg.norm(dv_p, axis=1)
            sigma2_np = sigma2.numpy()
            # confidence in standardized space — both numerator and denominator must match
            dv_std_mag = np.linalg.norm(dv_p / DV_STD, axis=1)
            conf = dv_std_mag / (np.sqrt(sigma2_np.mean(axis=1)) + 1e-8)

            for i in range(len(s)):
                gm = np.linalg.norm(dv_gt[i])
                pm = pred_mag[i]
                pred_mags.append(pm); gt_mags.append(gm); confidences.append(conf[i])
                if gm > 1e-8 and pm > 1e-8:
                    cos = float(np.dot(dv_p[i], dv_gt[i]) /
                                (np.linalg.norm(dv_p[i]) * gm))
                    cos_sims.append(cos)
                seq_errors[seq].append((np.sum((dv_p[i]-dv_gt[i])**2), gm))

            r2_num += float(((dv_p - dv_gt)**2).sum())
            r2_den += np.sum((dv_gt - dv_gt.mean(0))**2)

    pred_mags   = np.array(pred_mags)
    gt_mags     = np.array(gt_mags)
    cos_sims    = np.array(cos_sims)
    confidences = np.array(confidences)
    gate_fires  = confidences > DV_CONF_THRESHOLD
    r2          = 1.0 - r2_num / max(r2_den, 1e-8)

    print(f'\n  ── A. Overall metrics')
    print(f'     R²           : {r2:.4f}')
    print(f'     CosSim mean  : {cos_sims.mean():.4f}  (>0.5: strong)')
    print(f'     CosSim <0    : {(cos_sims < 0).mean()*100:.1f}%  (wrong direction)')
    print(f'     Conf gate fires: {gate_fires.mean()*100:.1f}%  (threshold={DV_CONF_THRESHOLD})')
    print(f'     Mean confidence: {confidences.mean():.3f}  std={confidences.std():.3f}')

    print(f'\n  ── B. Magnitude calibration by speed bin')
    print(f'     {"Speed bin":<20} {"n":>6} {"pred/gt":>8} {"CosSim":>8} {"conf_gate%":>10}')
    print(f'     {"-"*54}')
    for lo, hi in [(0, 0.001), (0.001, 0.003), (0.003, 0.01), (0.01, 0.05), (0.05, 1.0)]:
        mask = (gt_mags >= lo) & (gt_mags < hi)
        if mask.sum() < 5: continue
        ratio    = (pred_mags[mask] / np.maximum(gt_mags[mask], 1e-9)).mean()
        cs_mean  = cos_sims[mask[:len(cos_sims)]].mean() if len(cos_sims) > 0 else 0.0
        gp       = gate_fires[mask].mean() * 100
        print(f'     [{lo:.3f},{hi:.3f})        {mask.sum():>6d} {ratio:>8.3f} '
              f'{cs_mean:>8.3f} {gp:>9.1f}%')

    print(f'\n  ── C. Per-sequence quality (best 5 / worst 5)')
    seq_summary = {}
    for seq, errs in seq_errors.items():
        rel = np.sqrt(np.mean([e[0] for e in errs])) / max(np.mean([e[1] for e in errs]), 1e-8)
        seq_summary[seq] = rel
    sorted_seqs = sorted(seq_summary.items(), key=lambda x: x[1])
    print(f'  Best  5:  ' + '  '.join(f'{s}={v:.2f}' for s, v in sorted_seqs[:5]))
    print(f'  Worst 5:  ' + '  '.join(f'{s}={v:.2f}' for s, v in sorted_seqs[-5:]))

    # V9: confidence gate sweep (replaces pixel threshold sweep)
    print(f'\n  ── D. Confidence gate sensitivity (V9)')
    print(f'     {"threshold":>10} {"fires%":>8} {"signal_suppressed%":>20}')
    for thresh in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        fired   = confidences > thresh
        supp    = gt_mags[~fired]
        lost    = (supp > gt_mags.mean()).mean() * 100 if len(supp) else 0
        print(f'     {thresh:>10.1f} {fired.mean()*100:>7.1f}% {lost:>19.1f}%')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Failure Mode Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

def analyze_failure_modes(res_dir, checkpoint=None):
    print(f'\n{SEP2}')
    print('  SECTION 6: FAILURE MODE TAXONOMY')
    print(SEP2)

    if not checkpoint:
        print('  No checkpoint — skipping (pass --checkpoint).'); return

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from gcmc import GCMC
        model = GCMC(); model.eval()
        ckpt  = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
    except Exception as e:
        print(f'  Cannot load model: {e}'); return

    frames = _load_all_residuals(res_dir, 'val', max_seqs=25)
    if not frames: return

    modes = defaultdict(int)
    total = 0

    with torch.no_grad():
        for s, dv_gt, seq in frames:
            if len(s) < 2: continue
            s_t = torch.tensor(s, dtype=torch.float32)
            dv_pred, sigma2, _ = model(s_t)
            dv_p = dv_pred.numpy() * DV_STD

            pred_mag  = np.linalg.norm(dv_p, axis=1)
            gt_mag    = np.linalg.norm(dv_gt, axis=1)
            sigma2_np = sigma2.numpy()
            # V9 confidence gate
            """change
            conf = pred_mag / (np.sqrt(sigma2_np.mean(axis=1)) + 1e-8)
            """
            dv_std_mag = np.linalg.norm(dv_p / DV_STD, axis=1)
            conf = dv_std_mag / (np.sqrt(sigma2_np.mean(axis=1)) + 1e-8)

            
            fired = conf > DV_CONF_THRESHOLD

            for i in range(len(s)):
                total += 1
                gm = gt_mag[i]; pm = pred_mag[i]

                if gm < 1e-4 and not fired[i]:
                    modes['TN: both near-zero'] += 1
                elif gm < 1e-4 and fired[i]:
                    modes['FP: spurious correction'] += 1
                elif gm >= 1e-4 and not fired[i]:
                    modes['FN: gate suppressed signal'] += 1
                else:
                    cos = float(np.dot(dv_p[i], dv_gt[i]) /
                                (np.linalg.norm(dv_p[i]) * gm + 1e-9))
                    mag_ratio = pm / max(gm, 1e-9)
                    if cos < 0:
                        modes['TP-wrong-dir: wrong direction'] += 1
                    elif mag_ratio < 0.2:
                        modes['TP-under: severe underestimation'] += 1
                    elif mag_ratio > 5.0:
                        modes['TP-over: severe overestimation'] += 1
                    elif cos > 0.5:
                        modes['TP-good: correct direction+scale'] += 1
                    else:
                        modes['TP-partial: weak alignment'] += 1

    print(f'\n  Total predictions: {total:,}')
    print(f'\n  {"Mode":<45} {"count":>8} {"pct":>7}')
    print(f'  {"-"*62}')
    for mode, cnt in sorted(modes.items(), key=lambda x: -x[1]):
        print(f'  {mode:<45} {cnt:>8,} {100*cnt/total:>6.1f}%')

    tp_good   = modes.get('TP-good: correct direction+scale', 0)
    fp_spur   = modes.get('FP: spurious correction', 0)
    fn_gate   = modes.get('FN: gate suppressed signal', 0)
    wrong_dir = modes.get('TP-wrong-dir: wrong direction', 0)

    print(f'\n  V9 gate effectiveness vs V7 baseline:')
    print(f'  TP-good target: >31.0% (V7 baseline)')
    print(f'  FN target     : <16.0% (V7 baseline)')
    print(f'  Wrong-dir target: <26.6% (V7 baseline)')
    print(f'  FP target     : <9.0% (V7 baseline)')
    print()
    if fp_spur / max(total, 1) > 0.09:
        print(f'  ⚠ FP {100*fp_spur/total:.1f}% above V7 baseline — confidence gate too loose')
    if fn_gate / max(total, 1) > 0.16:
        print(f'  ⚠ FN {100*fn_gate/total:.1f}% above V7 baseline — confidence gate too tight')
    if wrong_dir / max(total, 1) > 0.266:
        print(f'  ⚠ Wrong-dir {100*wrong_dir/total:.1f}% above V7 — V9 feature changes not helping direction')
    if tp_good / max(total, 1) > 0.31:
        print(f'  ✓ TP-good {100*tp_good/total:.1f}% exceeds V7 baseline')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Improvement Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(res_dir, log_path=None, checkpoint=None):
    print(f'\n{SEP2}')
    print('  SECTION 7: STATUS & RECOMMENDATIONS')
    print(SEP2)

    print(f'\n  V9 implemented changes:')
    print(f'  ✓ [ARCH]  pos_spread replaces group_vel (dims 8-9)')
    print(f'  ✓ [ARCH]  own_vel → vel_deviation from group (dims 16-19)')
    print(f'  ✓ [TRAIN] Phase B freezes backbone — uncertainty head only')
    print(f'  ✓ [INFRA] Confidence gate replaces hard pixel threshold')

    issues = []

    if log_path and Path(log_path).exists():
        with open(log_path) as f:
            log = json.load(f)
        corr_rs = [e.get('tr_corr_ratio', e.get('corr_ratio', 0)) for e in log]
        phases  = [e.get('phase', 'A') for e in log]
        a_vals  = [c for c, p in zip(corr_rs, phases) if p == 'A']
        b_vals  = [c for c, p in zip(corr_rs, phases) if p == 'B']

        if a_vals and max(a_vals) < 0.40:
            issues.append(('ARCH', 'CorrR still below 0.40',
                f'Peak CorrR={max(a_vals):.4f}. V9 feature changes may not have broken '
                f'the SNR ceiling. If still at ~0.33, the remaining lever is input '
                f'noise reduction: whiten dv_star targets per-sequence before training.'))
        if b_vals and a_vals and b_vals[-1] < a_vals[-1] - 0.003:
            issues.append(('TRAIN', 'Phase B still eroding CorrR',
                f'Backbone freeze (Change 2) should have fixed this. '
                f'If still degrading, verify optimizer_B only covers uncertainty_head.parameters().'))

    issues.extend([
        ('ARCH', 'SNR ceiling — target whitening',
         'Dataset SNR=4.5x is the hard limit. To push CorrR > 0.40: '
         'normalise dv_star targets per-sequence (zero-mean, unit-std per seq). '
         'This removes sequence-level bias from targets without changing the model.'),

        ('ARCH', 'Attention collapse risk',
         'With DanceTrack group sizes p95=24, softmax attention over 24 neighbours '
         'with tau=0.05 may collapse to top-1. Monitor attention entropy in §2. '
         'If entropy < 0.5, increase tau_init to 0.10.'),

        ('EVAL', 'SORT exclusion',
         'SORT degrades by design (no robust re-matching). '
         'Exclude from primary evaluation — report ByteTrack and OC-SORT only.'),
    ])

    priority = {'ARCH': '🔴 HIGH', 'TRAIN': '🟡 MED', 'INFRA': '🟡 MED', 'EVAL': '🟢 LOW'}

    print(f'\n  Remaining issues:')
    print(f'\n  {"#":<3} {"Priority":<10} {"Category":<8} {"Issue"}')
    print(f'  {"-"*70}')
    for i, (cat, title, _) in enumerate(issues, 1):
        print(f'  {i:<3} {priority[cat]:<10} {cat:<8} {title}')

    print(f'\n  Detail:')
    for i, (cat, title, detail) in enumerate(issues, 1):
        print(f'\n  [{i}] {priority[cat]} {cat}: {title}')
        words = detail.split(); line = '      '
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line); line = '      ' + w + ' '
            else:
                line += w + ' '
        if line.strip(): print(line)

    print(f'\n  {SEP}')
    print(f'  V9 gate checks (eval after ep40):')
    print(f'     CorrR     > 0.38  (V7 peak: 0.3317)')
    print(f'     R²        > 0.53')
    print(f'     CosSim    > 0.22  (V7: 0.200)')
    print(f'     TP-good   > 31%   (V7 baseline)')
    print(f'     FP rate   < 9%    (V7 baseline)')
    print(f'     Wrong-dir < 26.6% (V7 baseline)')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--residuals',  default='residuals')
    p.add_argument('--log',        default=None)
    p.add_argument('--log2',       default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--mode',       default='full',
                   choices=['data','model','log','attribution','quality',
                             'failures','recommend','full','report','compare'])
    args = p.parse_args()

    res     = Path(args.residuals)
    run_all = args.mode in ('full', 'report')

    if run_all or args.mode == 'data':
        analyze_data(res)
    if run_all or args.mode == 'model':
        analyze_model(res)
    if (run_all or args.mode == 'log') and args.log:
        analyze_log(Path(args.log), label=args.log)
    if run_all or args.mode == 'attribution':
        analyze_feature_attribution(res, checkpoint=args.checkpoint)
    if run_all or args.mode == 'quality':
        analyze_correction_quality(res, checkpoint=args.checkpoint)
    if run_all or args.mode == 'failures':
        analyze_failure_modes(res, checkpoint=args.checkpoint)
    if run_all or args.mode in ('recommend', 'report'):
        generate_recommendations(res, log_path=args.log, checkpoint=args.checkpoint)
    if args.mode == 'compare' and args.log and args.log2:
        analyze_log(Path(args.log),  label='v_prev')
        analyze_log(Path(args.log2), label='v_new')

    print(f'\n{SEP2}\n')


if __name__ == '__main__':
    main()
