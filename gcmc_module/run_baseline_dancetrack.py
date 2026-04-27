# run_baseline_dancetrack.py
import json
import numpy as np
import supervision as sv
from pathlib import Path
from trackers import SORTTracker, ByteTrackTracker, OCSORTTracker
from trackers.eval import evaluate_mot_sequences

DANCETRACK_VAL = Path("/data/pos+mot/Datadir/DanceTrack/val")
RESULTS_DIR    = Path("baseline_results")
TRACKERS = {
    "sort":      SORTTracker,
    "bytetrack": ByteTrackTracker,
    "ocsort":    OCSORTTracker,
}

def gt_to_oracle_detections(gt_path: Path) -> dict[int, np.ndarray]:
    """Load GT file, return {frame: xyxy_array} as oracle detections."""
    data = np.loadtxt(gt_path, delimiter=',')
    if data.ndim == 1:
        data = data[None]
    out = {}
    for frame in np.unique(data[:, 0].astype(int)):
        rows = data[data[:, 0] == frame]
        x, y, w, h = rows[:,2], rows[:,3], rows[:,4], rows[:,5]
        xyxy = np.stack([x, y, x+w, y+h], axis=1)
        out[frame] = xyxy
    return out

def track_sequence(tracker_cls, gt_path: Path, out_path: Path):
    """Run tracker on oracle detections, write MOT output."""
    oracle = gt_to_oracle_detections(gt_path)
    tracker = tracker_cls()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame in sorted(oracle.keys()):
        xyxy = oracle[frame].astype(np.float32)
        conf = np.ones(len(xyxy), dtype=np.float32)
        dets = sv.Detections(xyxy=xyxy, confidence=conf)
        result = tracker.update(dets)
        for i, tid in enumerate(result.tracker_id):
            if tid < 0:
                continue
            x1,y1,x2,y2 = result.xyxy[i]
            w, h = x2-x1, y2-y1
            lines.append(f"{frame},{tid},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,-1,-1,-1")
    out_path.write_text("\n".join(lines))

# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {}

for name, cls in TRACKERS.items():
    print(f"\n{'='*50}\nTracker: {name.upper()}\n{'='*50}")
    tracker_dir = RESULTS_DIR / name
    tracker_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted(DANCETRACK_VAL.iterdir()):
        gt_path  = seq_dir / "gt" / "gt.txt"
        out_path = tracker_dir / f"{seq_dir.name}.txt"
        if not gt_path.exists():
            continue
        print(f"  tracking {seq_dir.name}...", end=" ", flush=True)
        track_sequence(cls, gt_path, out_path)
        print("done")

    result = evaluate_mot_sequences(
        gt_dir=str(DANCETRACK_VAL),
        tracker_dir=str(tracker_dir),
        metrics=["CLEAR", "HOTA", "Identity"],
    )
    print(result.table(columns=["HOTA","AssA","DetA","MOTA","IDF1","Frag"]))
    all_results[name] = json.loads(result.aggregate.json())

# Save baseline reference
Path("baseline_dancetrack.json").write_text(json.dumps(all_results, indent=2))
print("\nSaved: baseline_dancetrack.json")