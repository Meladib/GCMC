"""
gcmc_mixin.py — Shared GCMC injection logic v2
===============================================
- DV_STD rescaling applied in predict_numpy (gcmc.py)
- Correction gated: only applied when |dv| > 0.5px
- Q_aug uses per-track Q with float64 casting
"""
from __future__ import annotations
import numpy as np
import torch
from gcmc_module.gcmc import GCMC, fanout_numpy, build_Q_aug

IM_W_DEFAULT = 1920.0
IM_H_DEFAULT = 1080.0
DV_MIN_PX    = 0.5   # ignore corrections smaller than this (noise gate)


class GCMCModule:
    def __init__(self, checkpoint: str, device: str = 'cpu'):
        self.device = device
        self.model  = GCMC().to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def apply(
        self,
        states_8d: list[np.ndarray],
        Q_bases:   list[np.ndarray],
        im_w: float = IM_W_DEFAULT,
        im_h: float = IM_H_DEFAULT,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        N = len(states_8d)
        if N < 2:
            return states_8d, Q_bases

        # Normalise
        scale       = np.array([im_w,im_h,im_w,im_h,im_w,im_h,im_w,im_h], dtype=np.float32)
        states_norm = np.stack(states_8d, axis=0).astype(np.float32) / scale  # (N,8)

        # Forward (dv already rescaled by DV_STD inside predict_numpy)
        dv_norm, sigma2 = self.model.predict_numpy(states_norm, device=self.device)

        # Pixel-space correction
        dv_px      = dv_norm * np.array([im_w, im_h], dtype=np.float32)  # (N,2)
        correction = fanout_numpy(dv_px)                                   # (N,8)

        # Noise gate: skip corrections smaller than DV_MIN_PX
        dv_mag = np.linalg.norm(dv_px, axis=1)   # (N,)
        mask   = dv_mag > DV_MIN_PX

        corrected = []
        for i in range(N):
            if mask[i]:
                corrected.append(states_8d[i] + correction[i])
            else:
                corrected.append(states_8d[i])

        # Q_aug — per-track, float64
        sigma2_px = sigma2 * np.array([im_w**2, im_h**2], dtype=np.float64)
        Q_augs = []
        for i in range(N):
            q = Q_bases[i].astype(np.float64)
            s = sigma2_px[i:i+1]
            Q_augs.append(build_Q_aug(q, s)[0])

        return corrected, Q_augs