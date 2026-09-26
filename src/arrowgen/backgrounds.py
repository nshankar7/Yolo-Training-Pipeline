"""Backgrounds: stock photos regraded to look submerged, and procedural water scenes.

Everything returned here is linear-light float32 BGR in [0, 1], along with a "veil" map:
the colour that haze tends towards at each pixel. Objects blend towards their local veil
with distance, which is what makes a far arrow sit in the scene instead of on top of it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .color import srgb_triplet_to_linear, srgb_u8_to_linear
from .config import BackgroundCfg, ProceduralCfg, StockCfg, uniform


def value_noise(rng: np.random.Generator, height: int, width: int, octaves: int) -> np.ndarray:
    """Multi-octave value noise in [-1, 1], built from upsampled random grids."""
    out = np.zeros((height, width), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = 2 + 2**o, 2 + 2**o
        grid = rng.uniform(-1.0, 1.0, size=(gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    return np.clip(out / max(total, 1e-6), -1.0, 1.0)


def procedural_background(
    rng: np.random.Generator, width: int, height: int, veil_lin: np.ndarray, cfg: ProceduralCfg
) -> np.ndarray:
    """A water column: bright near the surface, dark with depth, textured by noise."""
    power = uniform(rng, cfg.gradient_power)
    tilt = uniform(rng, cfg.gradient_tilt)
    top = uniform(rng, cfg.top_gain)
    bottom = uniform(rng, cfg.bottom_gain)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = (yy / max(height - 1, 1)) + tilt * (xx / max(width - 1, 1) - 0.5)
    t = np.clip(t, 0.0, 1.0) ** power
    gain = top + (bottom - top) * t

    octaves = int(rng.integers(cfg.noise_octaves[0], cfg.noise_octaves[1] + 1))
    amp = uniform(rng, cfg.noise_amplitude)
    gain *= 1.0 + amp * value_noise(rng, height, width, octaves)

    img = veil_lin[None, None, :] * gain[..., None]
    # A little per-channel wobble so the water is not perfectly uniform in hue.
    for c in range(3):
        img[..., c] *= 1.0 + 0.4 * amp * value_noise(rng, height, width, 2)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def stock_background(
    rng: np.random.Generator, img_u8: np.ndarray, width: int, height: int, cfg: StockCfg
) -> np.ndarray:
    """Random resized crop of a stock photo, converted to linear light."""
    h, w = img_u8.shape[:2]
    target_ratio = (width / height) * uniform(rng, cfg.crop_ratio)
    area = uniform(rng, cfg.crop_area) * h * w
    cw = min(w, np.sqrt(area * target_ratio))
    ch = min(h, cw / max(target_ratio, 1e-6))
    cw = min(w, ch * target_ratio)
    cw_i, ch_i = max(8, int(round(cw))), max(8, int(round(ch)))
    x0 = int(rng.integers(0, max(1, w - cw_i + 1)))
    y0 = int(rng.integers(0, max(1, h - ch_i + 1)))
    crop = img_u8[y0 : y0 + ch_i, x0 : x0 + cw_i]
    interp = cv2.INTER_AREA if (cw_i > width or ch_i > height) else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (width, height), interpolation=interp)
    if rng.random() < cfg.hflip_prob:  # backgrounds carry no labels, so this is always safe
        crop = np.ascontiguousarray(crop[:, ::-1])
    return srgb_u8_to_linear(crop)


def regrade_to_water(
    bg_lin: np.ndarray, beta_bgr: np.ndarray, water_lin: np.ndarray, distance_m: float
) -> np.ndarray:
    """Push a dry photo towards water colour, as if seen through `distance_m` of water."""
    t = np.exp(-beta_bgr * distance_m).astype(np.float32)
    return np.clip(bg_lin * t + water_lin * (1.0 - t), 0.0, 1.0).astype(np.float32)


def veil_map(bg_lin: np.ndarray, water_lin: np.ndarray, from_background: bool) -> np.ndarray:
    """Low-frequency haze colour per pixel.

    For water backgrounds the scene's own low frequencies are the veil. For stock photos the
    hue comes from the water type and only the brightness follows the photo, so a red car in
    the background does not tint the haze red.
    """
    small = cv2.resize(bg_lin, (16, 12), interpolation=cv2.INTER_AREA)
    low = cv2.resize(small, (bg_lin.shape[1], bg_lin.shape[0]), interpolation=cv2.INTER_LINEAR)
    if from_background:
        return np.clip(low, 1e-4, 1.0).astype(np.float32)
    lum = low.mean(axis=2, keepdims=True)
    modulation = np.clip(lum / max(float(lum.mean()), 1e-4), 0.5, 1.5)
    return np.clip(water_lin[None, None, :] * modulation, 1e-4, 1.0).astype(np.float32)


def water_colour(rng: np.random.Generator, veil_rgb_ranges) -> np.ndarray:
    """Sample the water's backscatter colour, returned as linear BGR."""
    rgb = [int(rng.integers(lo, hi + 1)) for lo, hi in veil_rgb_ranges]
    lin_rgb = srgb_triplet_to_linear(rgb)
    return np.array(lin_rgb[::-1], dtype=np.float32)


def make_background(
    rng: np.random.Generator,
    cfg: BackgroundCfg,
    width: int,
    height: int,
    water_lin: np.ndarray,
    beta_bgr: np.ndarray,
    stock_img: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (background, veil map, source name)."""
    if stock_img is None:
        bg = procedural_background(rng, width, height, water_lin, cfg.procedural)
        return bg, veil_map(bg, water_lin, from_background=True), "procedural"

    bg = stock_background(rng, stock_img, width, height, cfg.stock)
    regraded = rng.random() < cfg.stock.regrade_prob
    if regraded:
        d_bg = uniform(rng, cfg.stock.regrade_distance_m)
        bg = regrade_to_water(bg, beta_bgr, water_lin, d_bg)
    return bg, veil_map(bg, water_lin, from_background=False), "stock"
