from gcmc_module.gcmc_mixin import GCMCModule
from gcmc_module.sort_gcmc import SORTGCMCTracker
import supervision as sv
import numpy as np


def main():
    gcmc = GCMCModule('checkpoints_v7/gcmc_best.pt', device='cpu')
    tracker = SORTGCMCTracker(gcmc)

    # simulate 3 frames
    for frame in range(3):
        boxes = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400],
            [500, 100, 600, 200],
            [150, 300, 250, 400]
        ], dtype=np.float32)

        # add small noise
        boxes += np.random.randn(*boxes.shape) * 2

        detections = sv.Detections(
            xyxy=boxes,
            confidence=np.ones(4) * 0.9
        )

        results = tracker.update(detections)

        print(f'frame {frame}: {len(results.xyxy)} tracks')


if __name__ == "__main__":
    main()