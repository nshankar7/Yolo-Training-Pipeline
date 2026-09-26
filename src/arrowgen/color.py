"""sRGB <-> linear-light conversion.

Every compositing, water and sensor operation in the pipeline happens in linear light.
Images arrive and leave as 8-bit sRGB, so conversion happens exactly at those two edges.
"""

from __future__ import annotations

import numpy as np

_LINEAR_LUT_SIZE = 4096


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# uint8 sRGB -> linear float32
_SRGB_U8_TO_LINEAR = _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)

# linear float -> uint8 sRGB, via a quantised lookup table (faster than pow on megapixels)
_LINEAR_TO_SRGB_U8 = np.clip(
    np.round(_linear_to_srgb(np.linspace(0.0, 1.0, _LINEAR_LUT_SIZE, dtype=np.float64)) * 255.0),
    0,
    255,
).astype(np.uint8)


def srgb_u8_to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB image (any channel count) -> float32 linear in [0, 1]."""
    return _SRGB_U8_TO_LINEAR[img_u8]


def linear_to_srgb_u8(img: np.ndarray) -> np.ndarray:
    """float32 linear image -> uint8 sRGB."""
    idx = np.clip(img, 0.0, 1.0) * (_LINEAR_LUT_SIZE - 1)
    return _LINEAR_TO_SRGB_U8[idx.astype(np.int32)]


def srgb_triplet_to_linear(rgb_0_255) -> np.ndarray:
    """An sRGB colour given as 0-255 RGB -> linear float32 RGB."""
    c = np.asarray(rgb_0_255, dtype=np.float64) / 255.0
    return _srgb_to_linear(c).astype(np.float32)
