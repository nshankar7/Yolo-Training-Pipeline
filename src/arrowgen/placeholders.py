"""Placeholder arrow artwork.

These stand in for the real turn-arrow designs so the pipeline can be built and tested
before the artwork exists. They also double as test fixtures. Replace the files under
assets/arrows/raw/<class>/ with the real designs and re-run `arrowgen ingest`.

Two source styles are produced on purpose, because ingest must cope with both:
flat graphics on a solid background, and PNGs that already carry transparency.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

SS = 4  # supersampling factor for anti-aliased edges


def _scale(points: list[tuple[float, float]], size: int) -> np.ndarray:
    return np.array([[p[0] * size, p[1] * size] for p in points], dtype=np.float32)


def _draw_shaft(canvas: np.ndarray, pts: np.ndarray, thickness: float) -> None:
    cv2.polylines(
        canvas,
        [np.round(pts).astype(np.int32)],
        isClosed=False,
        color=255,
        thickness=max(1, int(round(thickness))),
        lineType=cv2.LINE_8,
    )


def _draw_head(canvas: np.ndarray, tip: np.ndarray, direction: np.ndarray, size: float) -> None:
    d = direction / (np.linalg.norm(direction) + 1e-9)
    perp = np.array([-d[1], d[0]])
    base = tip - d * size
    tri = np.array([tip, base + perp * size * 0.62, base - perp * size * 0.62], dtype=np.float32)
    cv2.fillPoly(canvas, [np.round(tri).astype(np.int32)], 255)


def loop_mask(
    size: int = 512, thickness_frac: float = 0.15, gap_deg: float = 95.0, mirrored: bool = False
) -> np.ndarray:
    """A circular 'loop' arrow: a ring with a gap and an arrowhead, like a rotate symbol.

    Rotating this design does not change which way it loops, but mirroring it does — which
    is exactly why its mirror is a separate class.
    """
    big = size * SS
    canvas = np.zeros((big, big), np.uint8)
    thickness = thickness_frac * big
    radius = 0.30 * big
    centre = (big * 0.5, big * 0.53)

    start = -90.0 + gap_deg / 2.0
    end = 270.0 - gap_deg / 2.0
    cv2.ellipse(
        canvas, (round(centre[0]), round(centre[1])), (round(radius), round(radius)),
        0, start, end, 255, max(1, round(thickness)),
    )
    # Arrowhead at the arc's start, sitting on the ring and pointing along it (counter-
    # clockwise on screen, where y runs downwards).
    a = math.radians(start)
    base = np.array([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)])
    tangent = np.array([math.sin(a), -math.cos(a)])
    head_len = thickness * 2.0
    _draw_head(canvas, base + tangent * head_len, tangent, head_len)

    if mirrored:
        canvas = np.ascontiguousarray(canvas[:, ::-1])
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def turn90_mask(size: int = 512, thickness_frac: float = 0.13) -> np.ndarray:
    """A 90-degree left turn: up the stem, then away to the left."""
    big = size * SS
    canvas = np.zeros((big, big), np.uint8)
    thickness = thickness_frac * big
    head = thickness * 1.9
    path = _scale([(0.64, 0.93), (0.64, 0.42), (0.30, 0.42)], big)
    _draw_shaft(canvas, path, thickness)
    _draw_head(canvas, np.array([0.10 * big, 0.42 * big]), np.array([-1.0, 0.0]), head)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def uturn_mask(size: int = 512, thickness_frac: float = 0.12) -> np.ndarray:
    """A U-turn to the left: up, around the top, then back down."""
    big = size * SS
    canvas = np.zeros((big, big), np.uint8)
    thickness = thickness_frac * big
    head = thickness * 1.9
    cx, cy, r = 0.50, 0.40, 0.20
    arc = [
        (cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a)))
        for a in range(0, 181, 6)
    ]
    path = _scale([(0.70, 0.90), (0.70, 0.40), *arc, (0.30, 0.40), (0.30, 0.64)], big)
    _draw_shaft(canvas, path, thickness)
    _draw_head(canvas, np.array([0.30 * big, 0.92 * big]), np.array([0.0, 1.0]), head)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _outlined(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Alpha of a slightly grown copy, used to paint a dark rim around the glyph."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(mask, k)


def flat_on_background(
    mask: np.ndarray, fill_bgr: tuple[int, int, int], bg_bgr: tuple[int, int, int]
) -> np.ndarray:
    """Flat artwork: coloured glyph with a dark rim, on a solid background."""
    h, w = mask.shape
    img = np.full((h, w, 3), bg_bgr, np.uint8)
    rim = _outlined(mask).astype(np.float32) / 255.0
    a = mask.astype(np.float32) / 255.0
    rim_col = np.array([30, 30, 30], np.float32)
    out = img.astype(np.float32)
    out = out * (1 - rim[..., None]) + rim_col * rim[..., None]
    out = out * (1 - a[..., None]) + np.array(fill_bgr, np.float32) * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def rgba_cutout(mask: np.ndarray, fill_bgr: tuple[int, int, int]) -> np.ndarray:
    """Transparent artwork: coloured glyph with a dark rim and a real alpha channel."""
    rim = _outlined(mask).astype(np.float32) / 255.0
    a = mask.astype(np.float32) / 255.0
    rim_col = np.array([30, 30, 30], np.float32)
    rgb = rim_col * np.ones((*mask.shape, 3), np.float32)
    rgb = rgb * (1 - a[..., None]) + np.array(fill_bgr, np.float32) * a[..., None]
    alpha = np.clip(rim * 255.0, 0, 255)
    return np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)


# Colours read off the example artwork: a dark navy glyph on a teal placard.
NAVY_BGR = (126, 69, 18)
TEAL_BGR = (211, 193, 47)

DESIGNS = ("loop", "turn90", "uturn")


def _write(path: Path, image: np.ndarray, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    written.append(path)


def write_placeholder_assets(raw_dir: Path, designs: tuple[str, ...] = ("loop",)) -> list[Path]:
    """Write placeholder raw artwork.

    The 'loop' design mimics the supplied example: a navy loop arrow on a teal square, saved
    as a flat graphic so ingest has to colour-key the background away — the same path real
    artwork of this kind will take.
    """
    raw_dir = Path(raw_dir)
    unknown = [d for d in designs if d not in DESIGNS]
    if unknown:
        raise ValueError(f"unknown design(s) {unknown}; choose from {list(DESIGNS)}")
    written: list[Path] = []

    if "loop" in designs:
        out = raw_dir / "loop_ccw"
        _write(out / "flat_navy_on_teal.png",
               flat_on_background(loop_mask(512, 0.15), NAVY_BGR, TEAL_BGR), written)
        _write(out / "flat_navy_on_teal_thick.png",
               flat_on_background(loop_mask(512, 0.19, gap_deg=64), NAVY_BGR, TEAL_BGR), written)
        _write(out / "cutout_navy.png", rgba_cutout(loop_mask(512, 0.15), NAVY_BGR), written)

    if "turn90" in designs:
        out = raw_dir / "turn90_left"
        _write(out / "flat_navy_on_white.png",
               flat_on_background(turn90_mask(512, 0.13), NAVY_BGR, (255, 255, 255)), written)
        _write(out / "flat_orange_on_white.png",
               flat_on_background(turn90_mask(512, 0.17), (30, 140, 245), (252, 250, 248)),
               written)

    if "uturn" in designs:
        out = raw_dir / "uturn_left"
        _write(out / "cutout_yellow.png", rgba_cutout(uturn_mask(512, 0.12), (40, 210, 250)),
               written)
        _write(out / "cutout_white.png", rgba_cutout(uturn_mask(512, 0.16), (245, 245, 245)),
               written)
    return written
