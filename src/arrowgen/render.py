"""Draw one posed arrow into the scene, or refuse to.

The contract this module keeps: when it returns success, the arrow is in the image *and*
visible enough to be labelled. When it fails, the canvas is untouched — every check runs
against a candidate composite, and only a passing candidate is written back. That invariant
is why the model is never asked to find an arrow that is not really there, and never shown
one it is told to ignore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .color import srgb_triplet_to_linear
from .config import AppearanceCfg, BoardCfg, VisibilityCfg, uniform
from .geometry import Projection, board_outline, local_scale, pyramid_level
from .registry import ArrowAsset

LUMA_RGB = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_BGR_SWAP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


@dataclass
class BoardSpec:
    """A generated sign board: the placard the arrow is printed on."""

    colour_bgr: np.ndarray  # linear BGR
    outline_asset: np.ndarray  # polygon in asset coordinates
    polygon_img: np.ndarray | None = None  # filled in once the pose is known


@dataclass
class ObjectAppearance:
    hue_deg: float
    saturation: float
    brightness: float
    illum_strength: float
    illum_angle_deg: float
    feather_px: float


@dataclass
class RenderResult:
    ok: bool
    reason: str = ""
    delta_e: float = 0.0
    level: int = 0
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


def sample_appearance(rng: np.random.Generator, cfg: AppearanceCfg) -> ObjectAppearance:
    return ObjectAppearance(
        hue_deg=uniform(rng, cfg.hue_shift_deg),
        saturation=uniform(rng, cfg.saturation),
        brightness=uniform(rng, cfg.brightness),
        illum_strength=uniform(rng, cfg.illumination_gradient),
        illum_angle_deg=float(rng.uniform(0.0, 360.0)),
        feather_px=uniform(rng, cfg.edge_feather_px),
    )


def sample_board(
    rng: np.random.Generator, cfg: BoardCfg, glyph_bbox: tuple[float, float, float, float]
) -> BoardSpec | None:
    """Decide whether this arrow is printed on a board, and what that board looks like."""
    if rng.random() >= cfg.prob or not cfg.palette:
        return None
    rgb = cfg.palette[int(rng.integers(len(cfg.palette)))]
    jitter = 1.0 + rng.uniform(-cfg.colour_jitter, cfg.colour_jitter, size=3)
    linear_rgb = np.clip(srgb_triplet_to_linear(rgb) * jitter.astype(np.float32), 0.0, 1.0)
    outline = board_outline(
        glyph_bbox,
        padding_frac=uniform(rng, cfg.padding_frac),
        square=cfg.square,
        corner_frac=uniform(rng, cfg.corner_frac),
    )
    return BoardSpec(colour_bgr=linear_rgb[::-1].copy(), outline_asset=outline)


def colour_matrix_bgr(hue_deg: float, saturation: float, brightness: float) -> np.ndarray:
    """A 3x3 linear colour transform: hue rotation, then saturation, then brightness.

    Linear operations commute with premultiplication, so this can be applied to the warped
    premultiplied pixels instead of re-processing the source asset for every sample.
    """
    a = math.radians(hue_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    third = 1.0 / 3.0
    sq = math.sqrt(third)
    hue = np.array(
        [
            [cos_a + (1 - cos_a) * third,
             third * (1 - cos_a) - sq * sin_a,
             third * (1 - cos_a) + sq * sin_a],
            [third * (1 - cos_a) + sq * sin_a,
             cos_a + third * (1 - cos_a),
             third * (1 - cos_a) - sq * sin_a],
            [third * (1 - cos_a) - sq * sin_a,
             third * (1 - cos_a) + sq * sin_a,
             cos_a + third * (1 - cos_a)],
        ]
    )
    sat = (1.0 - saturation) * np.outer(np.ones(3), LUMA_RGB) + saturation * np.eye(3)
    rgb = brightness * (sat @ hue)
    return (_BGR_SWAP @ rgb @ _BGR_SWAP).astype(np.float32)


def _illumination_gain(shape: tuple[int, int], strength: float, angle_deg: float) -> np.ndarray:
    h, w = shape
    if strength <= 1e-4:
        return np.ones((h, w, 1), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ux, uy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    half = max(math.hypot(w, h) / 2.0, 1e-6)
    t = ((xx - (w - 1) / 2.0) * ux + (yy - (h - 1) / 2.0) * uy) / half
    return np.clip(1.0 + strength * t, 0.0, 2.0)[..., None].astype(np.float32)


def _polygon_alpha(
    polygon: np.ndarray, origin: tuple[int, int], size: tuple[int, int], shift: int = 3
) -> np.ndarray:
    """Anti-aliased coverage of a polygon, in region coordinates."""
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    pts = (polygon - np.asarray(origin, dtype=np.float64)) * (1 << shift)
    cv2.fillPoly(
        mask, [np.round(pts).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=shift
    )
    return mask.astype(np.float32) / 255.0


def estimate_contrast(  # noqa: PLR0913
    canvas: np.ndarray,
    veil: np.ndarray,
    box: tuple[float, float, float, float],
    asset: ArrowAsset,
    appearance: ObjectAppearance,
    beta_bgr: np.ndarray,
    distance_m: float,
    board: BoardSpec | None = None,
) -> float:
    """Rough Lab distance between the arrow and whatever it would sit against.

    An approximation from mean colours, run during placement so a doomed pose can be
    resampled. With a board, the binding constraint is the weaker of "can the sign be seen
    against the scene" and "can the arrow be seen against its board". The exact per-pixel
    checks still run at render time.
    """
    height, width = canvas.shape[:2]
    x0 = max(0, min(width - 1, int(box[0])))
    y0 = max(0, min(height - 1, int(box[1])))
    x1 = max(x0 + 1, min(width, int(box[2]) + 1))
    y1 = max(y0 + 1, min(height, int(box[3]) + 1))
    bg = canvas[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    veil_local = veil[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    matrix = colour_matrix_bgr(appearance.hue_deg, appearance.saturation, appearance.brightness)
    glyph = np.clip(matrix @ asset.mean_colour(), 0.0, 1.0)
    t = np.exp(-beta_bgr * distance_m)
    seen_glyph = glyph * t + veil_local * (1.0 - t)

    if board is None:
        return _lab_distance(seen_glyph, bg)
    seen_board = board.colour_bgr * t + veil_local * (1.0 - t)
    return min(_lab_distance(seen_glyph, seen_board), _lab_distance(seen_board, bg))


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    pair = np.stack([a, b]).reshape(1, 2, 3).astype(np.float32)
    lab = cv2.cvtColor(np.clip(pair, 0, 1), cv2.COLOR_LBGR2Lab)
    return float(np.linalg.norm(lab[0, 0] - lab[0, 1]))


def mean_delta_e(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    """Mean CIELab distance the object introduced, over the pixels it covers."""
    if not mask.any():
        return 0.0
    lab_b = cv2.cvtColor(np.clip(before, 0, 1), cv2.COLOR_LBGR2Lab)
    lab_a = cv2.cvtColor(np.clip(after, 0, 1), cv2.COLOR_LBGR2Lab)
    diff = np.linalg.norm(lab_a[mask] - lab_b[mask], axis=1)
    return float(diff.mean())


def render_object(  # noqa: PLR0913 - a render call legitimately takes this much state
    canvas: np.ndarray,
    veil: np.ndarray,
    asset: ArrowAsset,
    projection: Projection,
    appearance: ObjectAppearance,
    beta_bgr: np.ndarray,
    distance_m: float,
    blur_k: float,
    visibility: VisibilityCfg,
    max_roi_area_frac: float,
    board: BoardSpec | None = None,
    glyph_contrast_min: float = 0.0,
) -> RenderResult:
    """Composite one arrow. On failure the canvas is restored and nothing is drawn."""
    height, width = canvas.shape[:2]
    x0, y0, x1, y1 = projection.box

    if (x1 - x0) * (y1 - y0) > max_roi_area_frac * width * height:
        return RenderResult(False, "roi_too_large")

    sigma = math.hypot(blur_k * distance_m, appearance.feather_px)
    pad = int(math.ceil(3.0 * sigma)) + 2
    rx0 = max(0, int(math.floor(x0)) - pad)
    ry0 = max(0, int(math.floor(y0)) - pad)
    rx1 = min(width, int(math.ceil(x1)) + pad + 1)
    ry1 = min(height, int(math.ceil(y1)) + pad + 1)
    if rx1 <= rx0 or ry1 <= ry0:
        return RenderResult(False, "outside_image")

    centroid = asset.hull.mean(axis=0)
    scale = local_scale(projection.h, centroid)
    levels = asset.pyramid()
    level = pyramid_level(scale, len(levels) - 1)
    src, a_k = levels[level]

    translate = np.array([[1.0, 0.0, -rx0], [0.0, 1.0, -ry0], [0.0, 0.0, 1.0]])
    h_roi = translate @ projection.h @ a_k
    roi = cv2.warpPerspective(
        src, h_roi, (rx1 - rx0, ry1 - ry0),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    glyph_rgb = np.ascontiguousarray(roi[..., :3])
    glyph_a = np.ascontiguousarray(roi[..., 3])[..., None]
    glyph_rgb = cv2.transform(
        glyph_rgb,
        colour_matrix_bgr(appearance.hue_deg, appearance.saturation, appearance.brightness),
    )
    gain = _illumination_gain(
        roi.shape[:2], appearance.illum_strength, appearance.illum_angle_deg
    )
    glyph_rgb *= gain
    np.clip(glyph_rgb, 0.0, glyph_a, out=glyph_rgb)  # premultiplied colour can't exceed alpha

    # The board sits behind the glyph and shares its lighting and its water.
    board_rgb = board_a = None
    if board is not None and board.polygon_img is not None:
        board_a = _polygon_alpha(
            board.polygon_img, (rx0, ry0), (rx1 - rx0, ry1 - ry0)
        )[..., None]
        board_rgb = board.colour_bgr[None, None, :] * board_a * gain
        sign_rgb = glyph_rgb + board_rgb * (1.0 - glyph_a)
        sign_a = glyph_a + board_a * (1.0 - glyph_a)
    else:
        sign_rgb, sign_a = glyph_rgb, glyph_a

    # Water between the sign and the camera: attenuate its light, add the veil's backscatter.
    t = np.exp(-beta_bgr * distance_m).astype(np.float32)
    veil_roi = veil[ry0:ry1, rx0:rx1]
    one_minus_t = (1.0 - t).astype(np.float32)
    sign_rgb = sign_rgb * t + sign_a * veil_roi * one_minus_t
    if board_rgb is not None:
        board_rgb = board_rgb * t + board_a * veil_roi * one_minus_t

    if sigma > 0.3:
        sign_rgb = cv2.GaussianBlur(sign_rgb, (0, 0), sigma)
        sign_a = cv2.GaussianBlur(sign_a[..., 0], (0, 0), sigma)[..., None]
        if board_rgb is not None:
            board_rgb = cv2.GaussianBlur(board_rgb, (0, 0), sigma)
            board_a = cv2.GaussianBlur(board_a[..., 0], (0, 0), sigma)[..., None]

    target = canvas[ry0:ry1, rx0:rx1]
    before = target.copy()
    composited = sign_rgb + before * (1.0 - sign_a)

    glyph_coverage = glyph_a[..., 0] > 0.5
    sign_coverage = sign_a[..., 0] > 0.5
    if glyph_coverage.sum() < 4:
        return RenderResult(False, "no_coverage")

    if board_rgb is not None:
        # Two separate questions: can the sign be seen at all, and can the arrow be told
        # apart from the board it is printed on?
        board_only = board_rgb + before * (1.0 - board_a)
        sign_visible = mean_delta_e(before, board_only, sign_coverage)
        glyph_visible = mean_delta_e(board_only, composited, glyph_coverage)
        if sign_visible < visibility.min_contrast_delta_e:
            return RenderResult(False, "low_contrast", delta_e=sign_visible)
        if glyph_visible < glyph_contrast_min:
            return RenderResult(False, "low_glyph_contrast", delta_e=glyph_visible)
        delta_e = min(sign_visible, glyph_visible)
    else:
        delta_e = mean_delta_e(before, composited, glyph_coverage)
        if delta_e < visibility.min_contrast_delta_e:
            return RenderResult(False, "low_contrast", delta_e=delta_e)

    target[...] = composited
    return RenderResult(True, "", delta_e=delta_e, level=level, roi=(rx0, ry0, rx1, ry1))
