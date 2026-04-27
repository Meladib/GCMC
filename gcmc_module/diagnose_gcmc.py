"""
diagnose_gcmc.py — Comprehensive GCMC Diagnostic (v4-aware)
============================================================
Updated for 20-dim architecture: neighbourhood (12) + own-state (8)

Usage:
    python diagnose_gcmc.py --residuals residuals --log checkpoints_v5/training_log.json --mode full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SEP  = '─' * 70
SEP2 = '═' * 70


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trend(values: list[float], window: int = 5) -> float:
    if len(values) < window:
        return 0.0
    x = np.arange(window)
    y = np.array(values[-window:])
    return float(np.polyfit(x, y, 1)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Target Characterization
# ─────────────────────────────────────────────────────────────────────────────

def analyze_data(res_dir: Path):
    print(f'\n{SEP2}')
    print('  SECTION 1: RESIDUAL DATA ANALYSIS')
    print(SEP2)

    for split in ['train', 'val']:
        split_dir = res_dir / split
        if not split_dir.exists():
            continue

        all_dv, all_states, all_vel, group_sizes = [], [], [], []
        n_zero_vel_frames = 0
        n_total_frames    = 0
        n_cold_frames     = 0

        for npz in sorted(split_dir.glob('*.npz')):
            d = np.load(npz, allow_pickle=True)
            for s, dv in zip(d['states_norm'], d['dv_star']):
                s  = np.array(s,  dtype=np.float32)
                dv = np.array(dv, dtype=np.float32)
                all_dv.append(dv)
                all_states.append(s)

                vc = ((s[:, 4] + s[:, 6]) / 2, (s[:, 5] + s[:, 7]) / 2)
                vel_mag = np.sqrt(vc[0]**2 + vc[1]**2)
                all_vel.extend(vel_mag.tolist())
                group_sizes.append(len(s))

                if vel_mag.max() < 1e-6:
                    n_zero_vel_frames += 1
                n_total_frames += 1

                if np.abs((s[:, 4] + s[:, 6]) / 2).max() < 1e-4:
                    n_cold_frames += 1

        dv_flat = np.concatenate(all_dv, axis=0)
        vel_arr  = np.array(all_vel)
        gs_arr   = np.array(group_sizes)
        dv_mag   = np.linalg.norm(dv_flat, axis=1)

        print(f'\n  [{split.upper()}]')
        print(f'  {SEP}')

        print(f'\n  ── A. Residual Δv* distribution')
        print(f'     Total samples      : {len(dv_flat):,}')
        print(f'     Mean |Δv*|         : {dv_mag.mean():.6f}  (norm)')
        print(f'     Std  |Δv*|         : {dv_mag.std():.6f}')
        print(f'     Median |Δv*|       : {np.median(dv_mag):.6f}')
        print(f'     p75  |Δv*|         : {np.percentile(dv_mag, 75):.6f}')
        print(f'     p95  |Δv*|         : {np.percentile(dv_mag, 95):.6f}')
        print(f'     Max  |Δv*|         : {dv_mag.max():.6f}')
        print(f'     % samples < 0.001  : {(dv_mag < 0.001).mean()*100:.1f}%')
        print(f'     % samples < 0.005  : {(dv_mag < 0.005).mean()*100:.1f}%')
        print(f'     % samples > 0.05   : {(dv_mag > 0.05).mean()*100:.1f}%')

        print(f'\n  ── B. KF velocity state distribution')
        print(f'     Mean |vc_KF|       : {vel_arr.mean():.6f}')
        print(f'     Median |vc_KF|     : {np.median(vel_arr):.6f}')
        print(f'     % frames with vc≈0 : {(vel_arr < 1e-6).mean()*100:.1f}%')
        print(f'     Frames ALL-zero-vel: {n_zero_vel_frames}/{n_total_frames} '
              f'({n_zero_vel_frames/max(n_total_frames,1)*100:.1f}%)')

        print(f'\n  ── C. Group size distribution')
        print(f'     Mean tracks/frame  : {gs_arr.mean():.1f}')
        print(f'     Median             : {np.median(gs_arr):.1f}')
        print(f'     % frames k<5       : {(gs_arr < 5).mean()*100:.1f}%')
        print(f'     % frames k≥10      : {(gs_arr >= 10).mean()*100:.1f}%')
        print(f'     Max                : {gs_arr.max()}')

        pct_cold = n_cold_frames / len(all_states) * 100
        print(f'\n  ── C2. Cold frame prevalence')
        print(f'     Cold frames (all vc≈0): {pct_cold:.1f}% of data')
        print(f'     {"⚠ High — MLP learning on dead features" if pct_cold > 20 else "✓ OK"}')

        # ── Feature informativeness ─────────────────────────────────
        print(f'\n  ── D. Feature informativeness (can f_i predict Δv*?)')
        
        # For v4: analyze both neighbour features AND own-state
        pos_spreads, vel_spreads, own_vel_mags, own_pos_mags = [], [], [], []
        for s, dv in zip(all_states[:500], all_dv[:500]):
            cx = (s[:, 0] + s[:, 2]) / 2
            cy = (s[:, 1] + s[:, 3]) / 2
            pos_spreads.append(np.std(cx) + np.std(cy))
            
            vx = (s[:, 4] + s[:, 6]) / 2
            vy = (s[:, 5] + s[:, 7]) / 2
            vel_spreads.append(np.std(vx) + np.std(vy))
            
            # Own-state: per-track velocity magnitude
            own_vel_mags.extend(np.sqrt(vx**2 + vy**2).tolist())
            own_pos_mags.extend(np.sqrt(cx**2 + cy**2).tolist())

        dv_sample = np.linalg.norm(np.concatenate(all_dv[:500]), axis=1)
        ps = np.array(pos_spreads)

        # Correlation: pos_spread vs |Δv*|
        if len(ps) == len(dv_sample[:len(ps)]):
            corr_pos = np.corrcoef(ps, dv_sample[:len(ps)])[0, 1]
            print(f'     Corr(pos_spread, |Δv*|): {corr_pos:.4f}  '
                  f'{"✓ informative" if abs(corr_pos) > 0.1 else "⚠ weak signal"}')

        # Correlation: vel_spread vs |Δv*|
        vs = np.array(vel_spreads)
        if len(vs) == len(dv_sample[:len(vs)]):
            corr_vel = np.corrcoef(vs, dv_sample[:len(vs)])[0, 1]
            print(f'     Corr(vel_spread, |Δv*|): {corr_vel:.4f}  '
                  f'{"✓ informative" if abs(corr_vel) > 0.1 else "⚠ weak signal"}')

        # NEW: Correlation: own-state velocity vs |Δv*| (the critical v4 signal)
        own_vel_arr = np.array(own_vel_mags[:len(dv_sample)])
        if len(own_vel_arr) == len(dv_sample):
            corr_own = np.corrcoef(own_vel_arr, dv_sample)[0, 1]
            print(f'     Corr(|own_vel|, |Δv*|):  {corr_own:.4f}  '
                  f'{"✓ CRITICAL SIGNAL for v4" if abs(corr_own) > 0.1 else "⚠ weak signal"}')

        print(f'     Mean vel spread in f_ij : {vs.mean():.6f}')
        print(f'     Mean pos spread in f_ij : {ps.mean():.6f}')
        print(f'     Vel/Pos ratio           : {vs.mean()/max(ps.mean(), 1e-8):.4f}')

        # ── SNR ─────────────────────────────────────────────────────
        noise_floor = np.percentile(dv_mag, 50)
        signal      = np.percentile(dv_mag, 95)
        snr = signal / max(noise_floor, 1e-8)
        print(f'\n  ── E. Learning difficulty (SNR)')
        print(f'     Noise floor (p50)  : {noise_floor:.6f}')
        print(f'     Signal (p95)       : {signal:.6f}')
        print(f'     SNR (p95/p50)      : {snr:.2f}x')
        if snr < 5:
            print(f'     ⚠ LOW SNR — sparse signal, zero-init dangerous')


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Model Architecture Analysis (v4-aware)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_model(res_dir: Path, log_path: Path | None = None):
    print(f'\n{SEP2}')
    print('  SECTION 2: MODEL ARCHITECTURE ANALYSIS')
    print(SEP2)

    npz = sorted((res_dir / 'train').glob('*.npz'))[0]
    d = np.load(npz, allow_pickle=True)

    # Find warm frame
    warm_frame = None
    for s, dv in zip(d['states_norm'], d['dv_star']):
        s = np.array(s, dtype=np.float32)
        vc_mag = np.abs((s[:, 4] + s[:, 6]) / 2).max()
        if vc_mag > 1e-4 and len(s) >= 3:
            warm_frame = (s, np.array(dv, dtype=np.float32))
            break

    cold_frame = (np.array(d['states_norm'][0], dtype=np.float32),
                  np.array(d['dv_star'][0],     dtype=np.float32))

    print(f'\n  ── A. Feature quality: cold frame (frame 0, all vc=0)')
    _analyze_features_v4(cold_frame[0], label='cold')

    if warm_frame:
        print(f'\n  ── B. Feature quality: warm frame (vc > 0)')
        _analyze_features_v4(warm_frame[0], label='warm')

    # Loss landscape
    print(f'\n  ── C. Loss landscape analysis')
    print(f'     NLL = sq_err/σ² + log(σ²)')
    print(f'     Minimum at σ²* = sq_err')

    if log_path and log_path.exists():
        log = json.loads(log_path.read_text())
        final_mse = log[-1].get('train_mse', 0)
        final_sharp = log[-1].get('tr_sharpness', log[-1].get('val_sharpness', 0))
        final_nll = log[-1].get('train_nll', 0)
        print(f'     Final train MSE ≈ {final_mse:.6f}')
        print(f'     Final σ² (sharpness) ≈ {final_sharp:.6f}')
        print(f'     Final NLL ≈ {final_nll:.4f}')
        if final_mse > 0:
            opt_nll = 1 + np.log(final_mse)
            print(f'     Optimal NLL at this MSE ≈ {opt_nll:.4f}')
            gap = final_nll - opt_nll
            print(f'     Gap to optimal: {gap:.4f}')
            # v4-specific: check if floor is the bottleneck
            floor_ratio = final_sharp / final_mse if final_mse > 0 else float('inf')
            print(f'     σ²/MSE ratio: {floor_ratio:.1f}x')
            if floor_ratio > 100:
                print(f'     ⚠ CRITICAL: σ² is {floor_ratio:.0f}× larger than optimal')
                print(f'        Fix: Lower softplus floor from +1e-2 to +1e-6')
            elif floor_ratio > 10:
                print(f'     ⚠ σ² inflated — check floor value')
            else:
                print(f'     ✓ σ² near optimal')
    else:
        print(f'     (No training log provided)')

    print(f'\n  ── D. Gradient flow problem (theoretical)')
    print(f'     NLL = error_term + uncertainty_term')
    print(f'     error_term = (dv_pred - dv*)² / σ²')
    print(f'     If σ² >> MSE, error_term gradients → 0')
    print(f'     Correction head starves regardless of feature quality')

    print(f'\n  ── E. Feature dimensionality check (v4)')
    print(f'     f_i ∈ R²⁰ = [μ_pos(2), μ_vel(2), σ_pos(2), σ_vel(2),')
    print(f'                  group_vel(2), vel_dev(2), own_state(8)]')
    print(f'     Own-state dims 12-19: direct KF velocity estimate')
    print(f'     This is the signal that breaks the bottleneck')

    # Gradient check
    print(f'\n  ── F. Empirical gradient race check')
    try:
        check_gradient_flow_v4(res_dir)
    except Exception as e:
        print(f'     Could not run: {e}')


def _analyze_features_v4(states: np.ndarray, label: str):
    """Analyze v4 20-dim features: neighbour (12) + own-state (8)."""
    N = len(states)
    pos = (states[:, :2] + states[:, 2:4]) / 2.0
    vel = (states[:, 4:6] + states[:, 6:8]) / 2.0

    # Neighbourhood features (same as before)
    dpos = pos[:, None] - pos[None, :]
    dvel = vel[:, None] - vel[None, :]
    f_ij = np.concatenate([dpos, dvel], axis=-1)

    tau = 0.05
    dist = np.sqrt((dpos**2).sum(-1) + 1e-8)
    np.fill_diagonal(dist, 1e9)
    logits = -dist / tau
    logits -= logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits)
    alpha /= alpha.sum(axis=1, keepdims=True)

    mu = np.einsum('ij,ijd->id', alpha, f_ij)
    diff2 = (f_ij - mu[:, None])**2
    sigma = np.sqrt(np.einsum('ij,ijd->id', alpha, diff2) + 1e-6)

    # v4 additions
    group_vel = vel.mean(axis=0, keepdims=True).repeat(N, axis=0)
    vel_dev = vel - group_vel

    f_neighbour = np.concatenate([mu, sigma, group_vel, vel_dev], axis=-1)  # (N, 12)
    f_i = np.concatenate([f_neighbour, states], axis=-1)  # (N, 20)

    print(f'     [{label}] N={N}')
    print(f'     Neighbour features (0-11): mean={np.abs(f_neighbour).mean():.6f}')
    print(f'       μ_pos active: {np.abs(f_neighbour[:, :2]).mean():.6f}')
    print(f'       μ_vel active: {np.abs(f_neighbour[:, 2:4]).mean():.6f}')
    print(f'       group_vel:    {np.abs(f_neighbour[:, 8:10]).mean():.6f}')
    print(f'       vel_dev:      {np.abs(f_neighbour[:, 10:12]).mean():.6f}')
    print(f'     Own-state (12-19): mean={np.abs(states).mean():.6f}')
    print(f'       own_pos:      {np.abs(states[:, :4]).mean():.6f}')
    print(f'       own_vel:      {np.abs(states[:, 4:8]).mean():.6f}')
    print(f'     → v4 adds 8 dims of direct track state to 12 dims of context')


def check_gradient_flow_v4(res_dir: Path):
    """Check gradients for v4 20-dim model."""
    from gcmc import GCMC, nll_loss

    npz = sorted((res_dir / 'train').glob('*.npz'))[0]
    d = np.load(npz, allow_pickle=True)

    states_t, dv_gt_t = None, None
    for s, dv in zip(d['states_norm'], d['dv_star']):
        s_arr = np.array(s, dtype=np.float32)
        if len(s_arr) >= 3:
            states_t = torch.tensor(s_arr, dtype=torch.float32)
            dv_gt_t = torch.tensor(np.array(dv, dtype=np.float32), dtype=torch.float32)
            break

    if states_t is None:
        print('     No suitable frame found')
        return

    model = GCMC(tau_init=0.05)
    dv_pred, sigma2, _ = model(states_t)
    loss = nll_loss(dv_pred, dv_gt_t, sigma2)

    model.zero_grad()
    loss.backward()

    corr_norm = sum(p.grad.norm().item() for p in model.mlp.correction_head.parameters() if p.grad is not None)
    unc_norm = sum(p.grad.norm().item() for p in model.mlp.uncertainty_head.parameters() if p.grad is not None)
    tau_grad = model.aggregation.log_tau.grad
    tau_norm = tau_grad.norm().item() if tau_grad is not None else 0.0

    # v4-specific: check own-state encoder gradients (if exists)
    own_grad = 0.0
    # Note: v4 doesn't have separate own encoder, but check backbone input grads
    if hasattr(model.mlp.backbone[0], 'weight') and model.mlp.backbone[0].weight.grad is not None:
        # Gradient magnitude on first layer weights for own-state dims (12-19)
        grad_full = model.mlp.backbone[0].weight.grad
        if grad_full.shape[1] >= 20:
            own_grad = grad_full[:, 12:].norm().item()
            neigh_grad = grad_full[:, :12].norm().item()

    ratio = corr_norm / max(unc_norm, 1e-12)

    print(f'     Loss on sample: {loss.item():.4f}')
    print(f'     correction_head grad:  {corr_norm:.6f}')
    print(f'     uncertainty_head grad: {unc_norm:.6f}')
    print(f'     tau grad:              {tau_norm:.6f}')
    if own_grad > 0:
        print(f'     own-state grad:        {own_grad:.6f}')
        print(f'     neighbour grad:        {neigh_grad:.6f}')
        print(f'     own/neigh ratio:       {own_grad/max(neigh_grad, 1e-12):.2f}')
    print(f'     corr/unc ratio:        {ratio:.4f}  '
          f'{"✓ balanced" if 0.1 < ratio < 10 else "⚠ RACE — correction starved" if ratio < 0.1 else "⚠ RACE — uncertainty starved"}')
    
    # v4-specific: diagnose floor impact
    print(f'     σ² on sample:          {sigma2.mean().item():.6f}')
    print(f'     MSE on sample:         {((dv_pred - dv_gt_t)**2).mean().item():.6f}')
    print(f'     σ²/MSE ratio:          {sigma2.mean().item() / max(((dv_pred - dv_gt_t)**2).mean().item(), 1e-8):.1f}x')
    print(f'     {"✓ σ² near optimal" if sigma2.mean().item() / max(((dv_pred - dv_gt_t)**2).mean().item(), 1e-8) < 10 else "⚠ σ² floor too high"}')


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Training Log Analysis (enhanced for v4)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_log(log_path: Path, label: str = 'run'):
    print(f'\n{SEP2}')
    print(f'  SECTION 3: TRAINING LOG — {label.upper()}')
    print(SEP2)

    log = json.loads(log_path.read_text())
    n = len(log)

    epochs = [e['epoch'] for e in log]
    train_nll = [e['train_nll'] for e in log]
    val_nll = [e['val_nll'] for e in log]
    train_mse = [e.get('train_mse', 0) for e in log]
    sharpness = [e.get('tr_sharpness', 0) for e in log]
    act_rate = [e.get('tr_activation_rate', 0) for e in log]
    corr_ratio = [e.get('tr_corr_ratio', 0) for e in log]
    err_term = [e.get('tr_error_term', 0) for e in log]
    unc_term = [e.get('tr_uncertainty_term', 0) for e in log]

    print(f'\n  ── A. Convergence summary ({n} epochs)')
    print(f'     {"Epoch":>6}  {"TrainNLL":>10}  {"ValNLL":>10} '
          f'{"TrainMSE":>10}  {"Sharpness":>10}  {"ActRate":>8}  {"CorrRatio":>10}')
    print(f'     {"-"*68}')
    show_epochs = sorted(set([1] + [e for e in [5,10,15,20,25,30,40,50,75,100] if e <= n] + [n]))
    for e in log:
        if e['epoch'] in show_epochs:
            print(f'     {e["epoch"]:>6}  {e["train_nll"]:>10.4f}  {e["val_nll"]:>10.4f}  '
                  f'{e.get("train_mse",0):>10.6f}  {e.get("tr_sharpness",0):>10.6f}  '
                  f'{e.get("tr_activation_rate",0):>8.4f}  {e.get("tr_corr_ratio",0):>10.4f}')

    print(f'\n  ── B. What actually improved?')
    nll_gain = train_nll[0] - train_nll[-1]
    mse_gain = (train_mse[0] - train_mse[-1]) / max(abs(train_mse[0]), 1e-8) * 100
    sharp_drop = (sharpness[0] - sharpness[-1]) / max(abs(sharpness[0]), 1e-8) * 100
    print(f'     NLL gain (e1→e{n})     : {nll_gain:+.4f}')
    print(f'     MSE change             : {mse_gain:+.2f}%  '
          f'{"✓ improved" if mse_gain > 1 else "⚠ FLAT — correction head not learning"}')
    print(f'     Sharpness change       : {sharp_drop:+.1f}%  '
          f'{"⚠ σ² exploded" if sharp_drop < -50 else "✓ OK" if abs(sharp_drop) < 50 else "⚠ σ² collapsed"}')

    # NLL decomposition
    if err_term[0] != 0 and unc_term[0] != 0:
        et_change = err_term[-1] - err_term[0]
        ut_change = unc_term[-1] - unc_term[0]
        total_nll_change = (train_nll[-1] - train_nll[0])
        et_pct = et_change / max(abs(total_nll_change), 1e-8) * 100
        ut_pct = ut_change / max(abs(total_nll_change), 1e-8) * 100
        print(f'\n  ── C. NLL decomposition: error_term vs uncertainty_term')
        print(f'     error_term change     : {et_change:+.6f}  ({et_pct:+.1f}% of NLL Δ)')
        print(f'     uncertainty_term Δ    : {ut_change:+.6f}  ({ut_pct:+.1f}% of NLL Δ)')
        if abs(ut_pct) > 80 and abs(et_pct) < 20:
            print(f'     ⚠ CRITICAL: {abs(ut_pct):.0f}% of NLL gain = σ² gaming')
            print(f'       Correction head contributed NOTHING')
            print(f'       Fix: Lower σ² floor or harder phase separation')

    # Peak detection
    peak_idx = np.argmax(corr_ratio)
    peak_epoch = epochs[peak_idx]
    peak_corr = corr_ratio[peak_idx]
    print(f'\n  ── D. Activation analysis')
    print(f'     Peak CorrR: {peak_corr:.4f} at epoch {peak_epoch}')
    if peak_epoch < n - 10:
        print(f'     ⚠ CorrR peaked early then decayed — NLL phase killed learning')
        print(f'       Post-peak decline: {peak_corr - corr_ratio[-1]:.4f}')
    else:
        print(f'     ✓ CorrR still climbing or stable at end')
    print(f'     Final corr_ratio: {corr_ratio[-1]:.4f}')
    print(f'     Ideal corr_ratio: > 0.5')

    # Overfitting
    print(f'\n  ── E. Overfitting check')
    gap = [v - t for v, t in zip(val_nll, train_nll)]
    print(f'     Val-Train NLL gap: e1={gap[0]:+.4f} e{n//2}={gap[n//2-1]:+.4f} e{n}={gap[-1]:+.4f}')
    print(f'     {"⚠ overfitting" if gap[-1] > 0.5 else "✓ no overfitting"}')

    # Trends
    print(f'\n  ── G. Trend velocity (last 5 epochs)')
    nll_slope = _trend(train_nll)
    corr_slope = _trend(corr_ratio)
    print(f'     NLL slope:      {nll_slope:+.5f}/epoch  '
          f'{"✓ still learning" if nll_slope < -0.02 else "⚠ plateaued"}')
    print(f'     corr_ratio slope: {corr_slope:+.5f}/epoch  '
          f'{"✓ improving" if corr_slope > 0.001 else "⚠ flat"}')

    # v4-specific: σ²/MSE ratio trajectory
    if len(train_mse) > 0 and len(sharpness) > 0:
        print(f'\n  ── H. σ² calibration check')
        ratios = [s / max(m, 1e-8) for s, m in zip(sharpness, train_mse)]
        print(f'     σ²/MSE ratio: e1={ratios[0]:.1f} e{n//2}={ratios[n//2-1]:.1f} e{n}={ratios[-1]:.1f}')
        if ratios[-1] > 100:
            print(f'     ⚠ CRITICAL: σ² is {ratios[-1]:.0f}× larger than MSE')
            print(f'       Optimal ratio: ~1.0x')
            print(f'       Fix: Lower softplus floor from +1e-2 to +1e-6')

    # Epoch 50 gate
    if n >= 50:
        cr_50 = corr_ratio[49] if len(corr_ratio) > 49 else corr_ratio[-1]
        print(f'\n  ── I. Epoch-50 gate check')
        print(f'     corr_ratio@50: {cr_50:.4f}  {"✓ PASS >0.15" if cr_50 > 0.15 else "⚠ FAIL"}')
        if cr_50 <= 0.15:
            print(f'     → ARCHITECTURAL BOTTLENECK')
        elif cr_50 > 0.30 and corr_ratio[-1] < cr_50:
            print(f'     → CorrR peaked and decayed — fix σ² floor')
        else:
            print(f'     → Continue training')

    print(f'\n  ── F. Research verdict')
    if mse_gain < 1 and abs(sharp_drop) > 50:
        print(f'     ✗ DEGENERATE: σ² gaming without correction learning')
    elif peak_epoch < n - 10 and corr_ratio[-1] < peak_corr * 0.9:
        print(f'     ⚠ PEAK+DECAY: CorrR peaked at {peak_epoch} then collapsed')
        print(f'       Root cause: σ² floor too high (ratio={ratios[-1]:.0f}×)')
        print(f'       Fix: Lower floor to +1e-6, retrain from epoch {peak_epoch}')
    elif corr_ratio[-1] > 0.3:
        print(f'     ✓ HEALTHY: correction learning')
    else:
        print(f'     ~ PARTIAL: some learning but not converged')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', default='residuals')
    parser.add_argument('--log', default=None)
    parser.add_argument('--log2', default=None)
    parser.add_argument('--mode', default='full',
                        choices=['data', 'model', 'log', 'full', 'compare'])
    args = parser.parse_args()

    res = Path(args.residuals)

    if args.mode in ('data', 'full'):
        analyze_data(res)

    if args.mode in ('model', 'full'):
        log_p = Path(args.log) if args.log else None
        analyze_model(res, log_path=log_p)

    if args.mode in ('log', 'full') and args.log:
        analyze_log(Path(args.log), label=args.log)

    if args.mode == 'compare' and args.log and args.log2:
        analyze_log(Path(args.log), label='run1')
        analyze_log(Path(args.log2), label='run2')

    print(f'\n{SEP2}\n')


if __name__ == '__main__':
    main()
