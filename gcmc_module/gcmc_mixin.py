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
from gcmc_module_V10_claude.gcmc import GCMC, fanout_numpy, build_Q_aug

IM_W_DEFAULT = 1920.0
IM_H_DEFAULT = 1080.0
DV_CONF_THRESHOLD = 1   # fire if |dv_pred_norm| / sqrt(mean_sigma2) > this


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

        # Confidence gate: fire when prediction magnitude exceeds uncertainty
        
        dv_pred_mag = np.linalg.norm(dv_norm, axis=1)
        confidence  = dv_pred_mag / (np.sqrt(sigma2.mean(axis=1)) + 1e-8)
        """
        DV_STD = np.array([0.004789, 0.006376])
        dv_norm_std = dv_norm / DV_STD          # back to standardized space
        dv_std_mag  = np.linalg.norm(dv_norm_std, axis=1)
        confidence  = dv_std_mag / (np.sqrt(sigma2.mean(axis=1)) + 1e-8)
        """

        mask        = confidence > DV_CONF_THRESHOLD
        
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
