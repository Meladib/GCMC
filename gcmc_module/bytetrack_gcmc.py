"""
bytetrack_gcmc.py — ByteTrack + GCMC
======================================
Same injection pattern as SORT. ByteTrack has a two-stage association
but both stages use the same predicted states — GCMC corrects before both.
"""
from __future__ import annotations
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker
from gcmc_mixin import GCMCModule


class ByteTrackGCMCTracker(ByteTrackTracker):

    def __init__(self, gcmc_module: GCMCModule,
                 im_w: float = 1920., im_h: float = 1080., **kwargs):
        super().__init__(**kwargs)
        self.gcmc  = gcmc_module
        self.im_w  = im_w
        self.im_h  = im_h
        self._gcmc_pending = False

    def update(self, detections: sv.Detections) -> sv.Detections:
        pre_existing = list(self.tracks)

        for t in pre_existing:
            t.predict()

        active = [t for t in pre_existing if t.time_since_update == 1]
        if len(active) >= 2:
            states  = [t.state[:, 0].astype(np.float32) for t in active]
            q_bases = [t.Q for t in active]
            corrected, q_augs = self.gcmc.apply(states, q_bases, self.im_w, self.im_h)
            for i, t in enumerate(active):
                t.state[:, 0] = corrected[i]
                t.Q           = q_augs[i]

        saved = {id(t): t.predict for t in pre_existing}
        for t in pre_existing:
            t.predict = lambda: None

        result = super().update(detections)

        for t in self.tracks:
            if id(t) in saved:
                t.predict = saved[id(t)]

        return result