"""
eval_gcmc.py — Evaluate all 3 GCMC-augmented trackers vs baselines
===================================================================
Usage:
    python eval_gcmc.py \
        --dancetrack /data/pos+mot/Datadir/DanceTrack/val \
        --checkpoint checkpoints_v7/gcmc_best.pt \
        --out results_v7 \
        --device cpu

Outputs:
    results_v7/
        baseline_sort.txt ... bytetrack ... ocsort
        gcmc_sort.txt     ... bytetrack ... ocsort
        summary.json      ← delta table
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import supervision as sv

from trackers import SORTTracker, ByteTrackTracker, OCSORTTracker
from trackers.eval import evaluate_mot_sequences

from gcmc_mixin import GCMCModule
from sort_gcmc import SORTGCMCTracker
from bytetrack_gcmc import ByteTrackGCMCTracker
from ocsort_gcmc import OCSORTGCMCTracker


# ── GT → oracle detections ────────────────────────────────────────────────────
"""
def load_oracle_detections(gt_path: Path) -> dict[int, np.ndarray]:
    data = np.loadtxt(gt_path, delimiter=',')
    if data.ndim == 1: data = data[None]
    out = {}
    for frame in np.unique(data[:, 0].astype(int)):
        rows = data[data[:, 0] == frame]
        x, y, w, h = rows[:,2], rows[:,3], rows[:,4], rows[:,5]
        out[frame] = np.stack([x, y, x+w, y+h], axis=1).astype(np.float32)
    return out
"""
def load_oracle_detections(gt_path: Path) -> dict[int, np.ndarray]:
    data = np.loadtxt(gt_path, delimiter=',')
    if data.ndim == 1: data = data[None]
    # Filter: keep only conf=1 AND class=1 (MOT17 has ignore regions with conf=0)
    if data.shape[1] >= 7:
        data = data[(data[:, 6] == 1) & (data[:, 7] == 1)]
    out = {}
    for frame in np.unique(data[:, 0].astype(int)):
        rows = data[data[:, 0] == frame]
        x, y, w, h = rows[:,2], rows[:,3], rows[:,4], rows[:,5]
        out[frame] = np.stack([x, y, x+w, y+h], axis=1).astype(np.float32)
    return out



def read_seqinfo(seq_dir: Path) -> tuple[int, int, int]:
    """Returns (fps, im_w, im_h)."""
    fps, im_w, im_h = 20, 1920, 1080
    p = seq_dir / 'seqinfo.ini'
    if p.exists():
        for line in p.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip().lower()
                if k == 'framerate':  fps  = int(float(v))
                elif k == 'imwidth':  im_w = int(v)
                elif k == 'imheight': im_h = int(v)
    return fps, im_w, im_h


# ── Track one sequence ────────────────────────────────────────────────────────

def track_sequence(tracker, gt_path: Path, out_path: Path,
                   im_w: int, im_h: int) -> float:
    oracle = load_oracle_detections(gt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('') 
    lines = []
    t0 = time.perf_counter()
    for frame in sorted(oracle.keys()):
        xyxy = oracle[frame]
        conf = np.ones(len(xyxy), dtype=np.float32)
        dets = sv.Detections(xyxy=xyxy, confidence=conf)

        # Pass im_w/im_h if tracker supports it (GCMC variants)
        if hasattr(tracker, 'im_w'):
            tracker.im_w = im_w; tracker.im_h = im_h

        result = tracker.update(dets)
        if result.tracker_id is None or len(result.xyxy) == 0:
            continue
        for i, tid in enumerate(result.tracker_id):
            if tid is None or int(tid) < 0: continue
            x1,y1,x2,y2 = result.xyxy[i]
            lines.append(f'{frame},{tid},{x1:.1f},{y1:.1f},'
                         f'{x2-x1:.1f},{y2-y1:.1f},1,-1,-1,-1')
    fps_achieved = len(oracle) / max(time.perf_counter() - t0, 1e-6)
    out_path.write_text('\n'.join(lines))
    return fps_achieved


# ── Evaluate one tracker config ───────────────────────────────────────────────

def evaluate_tracker(name, tracker_factory, gt_dir, out_dir):
    tracker_dir = out_dir / name
    tracker_dir.mkdir(parents=True, exist_ok=True)
    fps_list = []

    # Track all sequences FIRST
    for seq_dir in sorted(gt_dir.iterdir()):
        gt_path = seq_dir / 'gt' / 'gt.txt'
        if not gt_path.exists(): continue
        _, im_w, im_h = read_seqinfo(seq_dir)
        tracker = tracker_factory()
        fps = track_sequence(tracker, gt_path,
                             tracker_dir / f'{seq_dir.name}.txt',
                             im_w, im_h)
        fps_list.append(fps)
        print(f'    {seq_dir.name}: {fps:.0f} fps')

    # Build seqmap from WRITTEN files only
    seqmap = out_dir / f'{name}_seqmap.txt'
    written = sorted(
        p.stem for p in tracker_dir.glob('*.txt')
        if p.stat().st_size > 0
    )
    seqmap.write_text('name\n' + '\n'.join(written) + '\n')

    # THEN evaluate
    result = evaluate_mot_sequences(
        gt_dir=str(gt_dir),
        tracker_dir=str(tracker_dir),
        seqmap=str(seqmap),
        metrics=['CLEAR', 'HOTA', 'Identity'],
    )

    """

    fps_list = []

    for seq_dir in sorted(gt_dir.iterdir()):
        gt_path = seq_dir / 'gt' / 'gt.txt'
        if not gt_path.exists(): continue
        _, im_w, im_h = read_seqinfo(seq_dir)

        tracker = tracker_factory()  # fresh instance per sequence
        fps = track_sequence(tracker, gt_path,
                             tracker_dir / f'{seq_dir.name}.txt',
                             im_w, im_h)
        fps_list.append(fps)
        print(f'    {seq_dir.name}: {fps:.0f} fps')

    result = evaluate_mot_sequences(
        gt_dir=str(gt_dir),
        tracker_dir=str(tracker_dir),
        metrics=['CLEAR', 'HOTA', 'Identity'],
    )
    """

    agg = result.aggregate
    metrics = {
        'HOTA':  round(agg.HOTA.HOTA  * 100, 3),
        'AssA':  round(agg.HOTA.AssA  * 100, 3),
        'DetA':  round(agg.HOTA.DetA  * 100, 3),
        'MOTA':  round(agg.CLEAR.MOTA * 100, 3),
        'IDF1':  round(agg.Identity.IDF1 * 100, 3),
        'IDSW':  int(agg.CLEAR.IDSW),
        'Frag':  int(agg.CLEAR.Frag),
        'mean_fps': round(float(np.mean(fps_list)), 1),
    }
    print(f'  [{name}] HOTA={metrics["HOTA"]} AssA={metrics["AssA"]} '
          f'MOTA={metrics["MOTA"]} IDSW={metrics["IDSW"]} '
          f'fps={metrics["mean_fps"]}')
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dancetrack', required=True)
    parser.add_argument('--checkpoint', default='checkpoints_v7/gcmc_best.pt')
    parser.add_argument('--out',        default='results_v7')
    parser.add_argument('--device',     default='cpu')
    args = parser.parse_args()

    gt_dir  = Path(args.dancetrack)
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    gcmc = GCMCModule(args.checkpoint, device=args.device)

    configs = {
        # Baselines
        'baseline_sort':      lambda: SORTTracker(),
        'baseline_bytetrack': lambda: ByteTrackTracker(),
        'baseline_ocsort':    lambda: OCSORTTracker(),
        # GCMC variants
        'gcmc_sort':          lambda: SORTGCMCTracker(gcmc),
        'gcmc_bytetrack':     lambda: ByteTrackGCMCTracker(gcmc),
        'gcmc_ocsort':        lambda: OCSORTGCMCTracker(gcmc),
    }

    all_results = {}
    for name, factory in configs.items():
        print(f'\n── {name} ──')
        all_results[name] = evaluate_tracker(name, factory, gt_dir, out_dir)

    # Delta table
    print('\n' + '═'*70)
    print('  DELTA TABLE (GCMC − baseline)')
    print('═'*70)
    print(f'  {"Tracker":<15} {"ΔHOTA":>7} {"ΔAssA":>7} '
          f'{"ΔMOTA":>7} {"ΔIDSW":>7} {"ΔFPS":>7}')
    print('  ' + '-'*60)
    for t in ['sort', 'bytetrack', 'ocsort']:
        b = all_results[f'baseline_{t}']
        g = all_results[f'gcmc_{t}']
        dH  = g['HOTA']  - b['HOTA']
        dA  = g['AssA']  - b['AssA']
        dM  = g['MOTA']  - b['MOTA']
        dI  = g['IDSW']  - b['IDSW']
        dF  = g['mean_fps'] - b['mean_fps']
        print(f'  {t:<15} {dH:>+7.3f} {dA:>+7.3f} '
              f'{dM:>+7.3f} {dI:>+7.0f} {dF:>+7.1f}')

    # Save
    all_results['deltas'] = {
        t: {
            'ΔHOTA': all_results[f'gcmc_{t}']['HOTA'] - all_results[f'baseline_{t}']['HOTA'],
            'ΔAssA': all_results[f'gcmc_{t}']['AssA'] - all_results[f'baseline_{t}']['AssA'],
            'ΔMOTA': all_results[f'gcmc_{t}']['MOTA'] - all_results[f'baseline_{t}']['MOTA'],
            'ΔIDSW': all_results[f'gcmc_{t}']['IDSW'] - all_results[f'baseline_{t}']['IDSW'],
        } for t in ['sort', 'bytetrack', 'ocsort']
    }
    (out_dir / 'summary.json').write_text(json.dumps(all_results, indent=2))
    print(f'\nSaved to {out_dir}/summary.json')


if __name__ == '__main__':
    main()