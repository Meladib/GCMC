"""
ocsort_gcmc.py — OC-SORT + GCMC
=================================
OC-SORT uses OCSORTTracklet with XYXYStateEstimator.
KF state accessed via: tracklet.kalman_filter.kf.x  (8,1)
                        tracklet.kalman_filter.kf.Q  (8,8)
"""
from __future__ import annotations
import numpy as np
import supervision as sv

from trackers import OCSORTTracker
from gcmc_mixin import GCMCModule


class OCSORTGCMCTracker(OCSORTTracker):

    def __init__(self, gcmc_module: GCMCModule,
                 im_w: float = 1920., im_h: float = 1080., **kwargs):
        super().__init__(**kwargs)
        self.gcmc  = gcmc_module
        self.im_w  = im_w
        self.im_h  = im_h

    def update(self, detections: sv.Detections) -> sv.Detections:
        pre_existing = list(self.tracks)

        # 1. Predict
        for tl in pre_existing:
            tl.predict()

        # 2. GCMC correction
        active = [tl for tl in pre_existing if tl.time_since_update <= 1]
        if len(active) >= 2:
            states  = [tl.kalman_filter.kf.x[:, 0].astype(np.float32) for tl in active]
            q_bases = [tl.kalman_filter.kf.Q for tl in active]
            corrected, q_augs = self.gcmc.apply(states, q_bases, self.im_w, self.im_h)
            for i, tl in enumerate(active):
                tl.kalman_filter.kf.x[:, 0] = corrected[i]
                tl.kalman_filter.kf.Q       = q_augs[i].astype(np.float64)

        # 3. Patch tracklet.predict (not kalman_filter.predict)
        saved = {id(tl): tl.predict for tl in pre_existing}
        for tl in pre_existing:
            tl.predict = lambda: None

        result = super().update(detections)

        for tl in self.tracks:
            if id(tl) in saved:
                tl.predict = saved[id(tl)]

        return result