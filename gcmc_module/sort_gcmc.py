"""
sort_gcmc.py — SORT + GCMC v2
==============================
Uses no-op predict patch (same as ByteTrack wrapper).
"""
from __future__ import annotations
import numpy as np
import supervision as sv

from trackers import SORTTracker
from gcmc_mixin import GCMCModule


class SORTGCMCTracker(SORTTracker):

    def __init__(self, gcmc_module: GCMCModule,
                 im_w: float = 1920., im_h: float = 1080., **kwargs):
        super().__init__(**kwargs)
        self.gcmc = gcmc_module
        self.im_w = im_w
        self.im_h = im_h

    def update(self, detections: sv.Detections) -> sv.Detections:
        pre_existing = list(self.trackers)

        for t in pre_existing:
            t.predict()

        if len(pre_existing) >= 2:
            states  = [t.state[:, 0].astype(np.float32) for t in pre_existing]
            q_bases = [t.Q for t in pre_existing]
            corrected, q_augs = self.gcmc.apply(states, q_bases, self.im_w, self.im_h)
            print(f'dv sample: {corrected[0][:4] - states[0][:4]}')
            for i, t in enumerate(pre_existing):
                t.state[:, 0] = corrected[i]
                t.Q           = q_augs[i]

        saved = {id(t): t.predict for t in pre_existing}
        for t in pre_existing:
            t.predict = lambda: None

        result = super().update(detections)

        for t in self.trackers:
            if id(t) in saved:
                t.predict = saved[id(t)]

        return result


