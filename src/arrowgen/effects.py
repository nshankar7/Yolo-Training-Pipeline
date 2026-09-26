"""Whole-frame camera and medium effects.

These are applied after compositing because they belong to the camera or the water, not to
any one object. None of them move an object, so labels stay valid.

Parameters are always sampled, whether or not the effect is enabled, so toggling one effect
off does not shift the random stream for the others. That is what makes an "effect strip"
(the same image with one effect removed) actually comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import EffectsCfg, uniform, uniform_int


@dataclass
class EffectParams:
    vignette: float | None
    defocus_sigma: float | None
    motion_len: int | None
    motion_angle_deg: float
    exposure_ev: float
    wb_gain: tuple[float, float]
    noise_shot: float | None
    noise_read: float
    resolution_factor: float | None
    extras: dict = field(default_factory=dict)


def sample_effects(rng: np.random.Generator, cfg: EffectsCfg) -> EffectParams:
    vig_on = rng.random() < cfg.vignetting.prob
    vig = uniform(rng, cfg.vignetting.strength)
    dof_on = rng.random() < cfg.defocus.prob
    dof = uniform(rng, cfg.defocus.sigma_px)
    mb_on = rng.random() < cfg.motion_blur.prob
    mb = uniform_int(rng, cfg.motion_blur.length_px)
    mb_angle = float(rng.uniform(0.0, 180.0))
    ev = uniform(rng, cfg.exposure_ev)
    wb = (uniform(rng, cfg.wb_gain), uniform(rng, cfg.wb_gain))
    noise_on = rng.random() < cfg.noise.prob
    shot = uniform(rng, cfg.noise.shot)
    read = uniform(rng, cfg.noise.read)
    res_on = rng.random() < cfg.resolution_loss.prob
    res = uniform(rng, cfg.resolution_loss.factor)
    return EffectParams(
        vignette=vig if vig_on else None,
        defocus_sigma=dof if dof_on else None,
        motion_len=mb if mb_on else None,
        motion_angle_deg=mb_angle,
        exposure_ev=ev,
        wb_gain=wb,
        noise_shot=shot if noise_on else None,
        noise_read=read,
        resolution_factor=res if res_on else None,
    )


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = max(3, length | 1)
    k = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for i in range(size):
        t = i - c
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    total = k.sum()
    return k / total if total > 0 else np.ones((1, 1), np.float32)


def vignette_gain(shape: tuple[int, int], strength: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - (w - 1) / 2.0) / max(w / 2.0, 1e-6)
    ny = (yy - (h - 1) / 2.0) / max(h / 2.0, 1e-6)
    r2 = np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)
    return (1.0 - strength * r2)[..., None].astype(np.float32)


def apply_effects(img: np.ndarray, p: EffectParams, rng: np.random.Generator) -> np.ndarray:
    """Apply optics, then sensor, then image-processing losses. Input/output: linear BGR."""
    out = img
    if p.vignette is not None:
        out = out * vignette_gain(out.shape[:2], p.vignette)
    if p.defocus_sigma is not None and p.defocus_sigma > 0.25:
        out = cv2.GaussianBlur(out, (0, 0), p.defocus_sigma)
    if p.motion_len is not None and p.motion_len >= 3:
        out = cv2.filter2D(out, -1, motion_blur_kernel(p.motion_len, p.motion_angle_deg))

    gain = np.array([p.wb_gain[0], 1.0, p.wb_gain[1]], np.float32) * (2.0**p.exposure_ev)
    out = out * gain

    if p.noise_shot is not None:
        sigma = np.sqrt(np.maximum(out, 0.0) * p.noise_shot + p.noise_read**2)
        out = out + rng.standard_normal(out.shape, dtype=np.float32) * sigma

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    if p.resolution_factor is not None:
        h, w = out.shape[:2]
        sw = max(8, int(round(w * p.resolution_factor)))
        sh = max(8, int(round(h * p.resolution_factor)))
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
