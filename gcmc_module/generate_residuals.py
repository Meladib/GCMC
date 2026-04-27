"""
Phase 2 — Residual Dataset Generation
======================================
For each train sequence:
  1. Replay GT tracks through the XYXY Kalman filter (predict + update)
  2. Compute center velocity from XYXY corners
  3. Compute residual: Δv* = vc_gt − vc_KF  (center space, normalised)
  4. Store per-frame: all active track states + residuals

Output:
  residuals/
    train/   {seq}.npz   ← used for MLP training
    val/     {seq}.npz   ← used for MLP validation
    norm.json            ← μ, σ for positions and velocities

Usage:
  python generate_residuals.py \
      --dancetrack /data/pos+mot/Datadir/DanceTrack \
      --out residuals \
      --val-ratio 0.2 \
      --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from trackers.utils.state_representations import XYXYStateEstimator


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt(gt_path: Path):
    """Return dict: track_id → sorted list of (frame, x1, y1, x2, y2)."""
    data = np.loadtxt(gt_path, delimiter=',')
    if data.ndim == 1:
        data = data[None]
    tracks = {}
    for row in data:
        frame = int(row[0])
        tid   = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        x1, y1, x2, y2 = x, y, x + w, y + h
        tracks.setdefault(tid, []).append((frame, x1, y1, x2, y2))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks

def read_seqinfo(seq_dir: Path):
    """Return (im_w, im_h) from seqinfo.ini."""
    im_w, im_h = 1920, 1080
    p = seq_dir / 'seqinfo.ini'
    if p.exists():
        for line in p.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                if k.strip().lower() == 'imwidth':  im_w = int(v)
                if k.strip().lower() == 'imheight': im_h = int(v)
    return im_w, im_h

def center_vel(state_8d):
    """Extract normalised center velocity from 8-dim XYXY state [x1,y1,x2,y2,vx1,vy1,vx2,vy2]."""
    vx = (state_8d[4] + state_8d[6]) / 2.0
    vy = (state_8d[5] + state_8d[7]) / 2.0
    return np.array([vx, vy], dtype=np.float64)

def center_pos(state_8d):
    """Extract center position from 8-dim XYXY state."""
    cx = (state_8d[0] + state_8d[2]) / 2.0
    cy = (state_8d[1] + state_8d[3]) / 2.0
    return np.array([cx, cy], dtype=np.float64)


# ── Per-sequence processing ───────────────────────────────────────────────────

def process_sequence(seq_dir: Path, im_w: int, im_h: int):
    """
    Returns list of frame records, each a dict:
      frame        : int
      track_ids    : (N,)   int
      states_norm  : (N, 8) float  — normalised XYXY states
      dv_star      : (N, 2) float  — residual Δv* in normalised coords
    Only frames with ≥2 active tracks are kept (kNN needs neighbours).
    """
    gt_path = seq_dir / 'gt' / 'gt.txt'
    gt = load_gt(gt_path)

    # Initialise one KF per track
    kf_map = {}   # tid → XYXYStateEstimator
    prev_center = {}  # tid → center at last update (for GT velocity)
    prev_frame  = {}  # tid → frame of last update

    # Collect all frames
    all_frames = sorted({r[0] for obs in gt.values() for r in obs})

    # Build frame → {tid: bbox} lookup
    frame_lookup = {}
    for tid, obs in gt.items():
        for (frame, x1, y1, x2, y2) in obs:
            frame_lookup.setdefault(frame, {})[tid] = np.array([x1, y1, x2, y2])

    records = []

    for frame in all_frames:
        active_tids = list(frame_lookup.get(frame, {}).keys())

        # ── Step 1: predict all existing KFs ──────────────────────────────
        predicted_states = {}
        for tid, kf in kf_map.items():
            kf.predict()
            predicted_states[tid] = kf.kf.x.flatten().copy()  # 8-dim

        # ── Step 2: compute residuals for tracks seen in this frame ───────
        tid_list, states_norm, dv_stars = [], [], []

        for tid in active_tids:
            bbox = frame_lookup[frame][tid]
            cx_gt = (bbox[0] + bbox[2]) / 2.0
            cy_gt = (bbox[1] + bbox[3]) / 2.0

            if tid not in kf_map:
                # First appearance — initialise KF, no residual yet
                kf_map[tid] = XYXYStateEstimator(bbox)
                prev_center[tid] = np.array([cx_gt, cy_gt])
                prev_frame[tid]  = frame
                continue

            # GT center velocity (finite difference)
            dt = frame - prev_frame[tid]
            if dt == 0:
                continue
            vc_gt = np.array([
                (cx_gt - prev_center[tid][0]) / dt,
                (cy_gt - prev_center[tid][1]) / dt,
            ])

            # KF predicted center velocity
            pred_state = predicted_states[tid]
            vc_kf = center_vel(pred_state)

            # Residual in pixel space, then normalise
            dv_star = vc_gt - vc_kf
            dv_star_norm = np.array([dv_star[0] / im_w, dv_star[1] / im_h])

            # Normalised state
            s = pred_state.copy()
            s[0] /= im_w; s[2] /= im_w; s[4] /= im_w; s[6] /= im_w
            s[1] /= im_h; s[3] /= im_h; s[5] /= im_h; s[7] /= im_h

            tid_list.append(tid)
            states_norm.append(s)
            dv_stars.append(dv_star_norm)

            # ── Step 3: update KF with GT observation ──────────────────────
            kf_map[tid].update(bbox)
            prev_center[tid] = np.array([cx_gt, cy_gt])
            prev_frame[tid]  = frame

        # Update KFs for tracks not seen this frame (predict only, no update)
        for tid in list(kf_map.keys()):
            if tid not in frame_lookup.get(frame, {}):
                kf_map[tid].update(None)

        if len(tid_list) >= 2:
            records.append({
                'frame':       frame,
                'track_ids':   np.array(tid_list, dtype=np.int32),
                'states_norm': np.array(states_norm, dtype=np.float32),
                'dv_star':     np.array(dv_stars,   dtype=np.float32),
            })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dancetrack', default='/data/pos+mot/Datadir/DanceTrack')
    parser.add_argument('--out',        default='residuals')
    parser.add_argument('--val-ratio',  type=float, default=0.2)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    root    = Path(args.dancetrack)
    out_dir = Path(args.out)
    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'val').mkdir(parents=True,   exist_ok=True)

    # Sequence-disjoint 80/20 split
    train_seqs = sorted((root / 'train').iterdir())
    random.seed(args.seed)
    random.shuffle(train_seqs)
    n_val  = max(1, int(len(train_seqs) * args.val_ratio))
    val_seqs   = train_seqs[:n_val]
    train_seqs = train_seqs[n_val:]

    print(f'Train seqs: {len(train_seqs)}  |  Val seqs: {len(val_seqs)}')

    # Collect all Δv* from train for normalisation stats
    all_dv = []

    for split_name, seq_list in [('train', train_seqs), ('val', val_seqs)]:
        for seq_dir in seq_list:
            if not (seq_dir / 'gt' / 'gt.txt').exists():
                continue
            im_w, im_h = read_seqinfo(seq_dir)
            print(f'  [{split_name}] {seq_dir.name}  ({im_w}x{im_h})', end=' ... ', flush=True)

            records = process_sequence(seq_dir, im_w, im_h)

            # Pack into arrays for efficient storage
            # Each .npz: frames, track_ids (ragged→object), states_norm (ragged), dv_star (ragged)
            frames      = np.array([r['frame']     for r in records], dtype=np.int32)
            track_ids   = np.array([r['track_ids'] for r in records], dtype=object)
            states_norm = np.array([r['states_norm'] for r in records], dtype=object)
            dv_star     = np.array([r['dv_star']   for r in records], dtype=object)

            npz_path = out_dir / split_name / f'{seq_dir.name}.npz'
            np.savez_compressed(npz_path,
                frames=frames,
                track_ids=track_ids,
                states_norm=states_norm,
                dv_star=dv_star,
                im_w=np.array([im_w]),
                im_h=np.array([im_h]),
            )

            n_samples = sum(len(r['dv_star']) for r in records)
            print(f'{len(records)} frames, {n_samples} samples')

            if split_name == 'train':
                for r in records:
                    all_dv.append(r['dv_star'])

    # ── Normalisation constants from train split ───────────────────────────
    all_dv_flat = np.concatenate(all_dv, axis=0)   # (M, 2)
    norm = {
        'dv_mean': all_dv_flat.mean(axis=0).tolist(),
        'dv_std':  all_dv_flat.std(axis=0).tolist(),
        'dv_max':  np.abs(all_dv_flat).max(axis=0).tolist(),
        'n_train_samples': int(len(all_dv_flat)),
        'val_seq_names': [s.name for s in val_seqs],
        'train_seq_names': [s.name for s in train_seqs],
    }
    (out_dir / 'norm.json').write_text(json.dumps(norm, indent=2))

    print(f'\nNormalisation (train Δv*):')
    print(f'  mean : {norm["dv_mean"]}')
    print(f'  std  : {norm["dv_std"]}')
    print(f'  |max|: {norm["dv_max"]}')
    print(f'  total train samples: {norm["n_train_samples"]:,}')
    print(f'\nSaved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
