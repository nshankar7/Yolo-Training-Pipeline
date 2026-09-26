"""Projective geometry: pose -> homography -> exact bounding box.

This module decides where an arrow lands in the image *and* what its label is, so the two
can never disagree. Conventions (OpenCV):

* Camera space: x right, y down, z forward (metres).
* Pixel coordinates: pixel centres sit at integer coordinates, so the image spans
  [-0.5, W-0.5] x [-0.5, H-0.5] in "pixel-edge" coordinates.
* An asset lives on the z=0 plane of its own frame; the sign's front face points along +z
  of that frame, i.e. towards the camera when it is readable.

The box is the min/max of the arrow's convex hull *after* projection. Using the hull rather
than the asset's four corners keeps boxes tight for L- and U-shaped arrows under roll and
perspective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

MIN_DEPTH_M = 0.05


@dataclass(frozen=True)
class Pose:
    """A planar sign's placement in front of the camera."""

    roll_deg: float
    yaw_deg: float
    pitch_deg: float
    distance_m: float
    center_px: tuple[float, float]
    size_m: float
    stretch: float


@dataclass(frozen=True)
class Projection:
    """Everything needed to draw a posed asset and label it."""

    h: np.ndarray  # 3x3 homography: asset pixel coords -> image pixel coords
    hull_img: np.ndarray  # (N, 2) float64, projected hull
    depths: np.ndarray  # (N,) float64, camera-space z of each hull point
    box: tuple[float, float, float, float]  # unclipped, pixel-edge coords


def camera_matrix(width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, float]:
    """Pinhole intrinsics from horizontal field of view."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    k = np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return k, f


def rotation_matrix(roll_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    r, y, p = (math.radians(v) for v in (roll_deg, yaw_deg, pitch_deg))
    cz, sz = math.cos(r), math.sin(r)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(p), math.sin(p)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def asset_to_plane(
    hull_bbox: tuple[float, float, float, float], size_m: float, stretch: float
) -> np.ndarray:
    """Map asset pixel coords to metres on the sign plane, centred on the glyph.

    `stretch` is an area-preserving aspect distortion: sx = sqrt(stretch), sy = 1/sqrt(stretch).
    `size_m` is the longest side of the stretched glyph in metres.
    """
    x0, y0, x1, y1 = hull_bbox
    cu, cv = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ew, eh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    sx = math.sqrt(stretch)
    sy = 1.0 / sx
    m = size_m / max(ew * sx, eh * sy)
    return np.array(
        [[m * sx, 0.0, -m * sx * cu], [0.0, m * sy, -m * sy * cv], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distance_for_apparent_size(
    f: float, size_m: float, apparent: float, long_side_px: int
) -> float:
    """Distance at which a sign of `size_m` covers `apparent` of the image's long side."""
    return f * size_m / max(apparent * long_side_px, 1e-9)


def build_homography(
    k: np.ndarray, pose: Pose, glyph_bbox: tuple[float, float, float, float]
) -> np.ndarray:
    """Compose K, R, t and the asset-to-plane map.

    Scale always refers to the *glyph's* bounding box, so `apparent_size` means the same
    thing whether or not a sign board is drawn around it.
    """
    r = rotation_matrix(pose.roll_deg, pose.yaw_deg, pose.pitch_deg)
    ray = np.linalg.inv(k) @ np.array([pose.center_px[0], pose.center_px[1], 1.0])
    t = pose.distance_m * ray  # t_z == distance because the ray has z == 1
    m = asset_to_plane(glyph_bbox, pose.size_m, pose.stretch)
    return k @ np.column_stack((r[:, 0], r[:, 1], t)) @ m


def project_outline(h: np.ndarray, points: np.ndarray) -> Projection:
    """Project a polygon given in asset coordinates through `h`."""
    pts = np.hstack([points, np.ones((len(points), 1))])  # (N, 3)
    proj = pts @ h.T
    depths = proj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = proj[:, :2] / depths[:, None]
    box = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
    return Projection(h=h, hull_img=q, depths=depths, box=box)


def build_projection(
    k: np.ndarray, pose: Pose, hull: np.ndarray, hull_bbox: tuple[float, float, float, float]
) -> Projection:
    """Convenience: build the homography and project one outline through it."""
    return project_outline(build_homography(k, pose, hull_bbox), hull)


def board_outline(
    glyph_bbox: tuple[float, float, float, float],
    padding_frac: float,
    square: bool,
    corner_frac: float,
    segments: int = 6,
) -> np.ndarray:
    """A sign board's outline in asset coordinates, as a (possibly rounded) rectangle."""
    x0, y0, x1, y1 = glyph_bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    if square:
        half_w = half_h = max(half_w, half_h)
    half_w *= 1.0 + 2.0 * padding_frac
    half_h *= 1.0 + 2.0 * padding_frac

    radius = corner_frac * min(half_w, half_h)
    if radius <= 1e-6:
        return np.array(
            [[cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
             [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]], dtype=np.float64
        )

    pts: list[tuple[float, float]] = []
    centres = [
        (cx + half_w - radius, cy + half_h - radius, 0.0),
        (cx - half_w + radius, cy + half_h - radius, 90.0),
        (cx - half_w + radius, cy - half_h + radius, 180.0),
        (cx + half_w - radius, cy - half_h + radius, 270.0),
    ]
    for ox, oy, start in centres:
        for i in range(segments + 1):
            a = math.radians(start + 90.0 * i / segments)
            pts.append((ox + radius * math.cos(a), oy + radius * math.sin(a)))
    return np.array(pts, dtype=np.float64)


def front_facing(k: np.ndarray, pose: Pose) -> tuple[bool, float]:
    """Is the sign's front towards the camera, and at what viewing angle (degrees)?

    det([r1 r2 t]) == r3 . t, and that is the sign of the projection's Jacobian determinant.
    Negative means we are looking at the back of the sign, where the arrow appears mirrored —
    which would silently be a different class. Such poses must be rejected.
    """
    r = rotation_matrix(pose.roll_deg, pose.yaw_deg, pose.pitch_deg)
    ray = np.linalg.inv(k) @ np.array([pose.center_px[0], pose.center_px[1], 1.0])
    t = pose.distance_m * ray
    dot = float(r[:, 2] @ t)
    if dot <= 0.0:
        return False, 180.0
    cos_theta = dot / float(np.linalg.norm(t))
    return True, math.degrees(math.acos(min(1.0, max(-1.0, cos_theta))))


def clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip a pixel-edge box to the image; None if nothing is left."""
    x0 = max(box[0], -0.5)
    y0 = max(box[1], -0.5)
    x1 = min(box[2], width - 0.5)
    y1 = min(box[3], height - 0.5)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def box_to_yolo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Clipped pixel-edge box -> normalised YOLO (cx, cy, w, h)."""
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0 + 0.5) / width
    cy = ((y0 + y1) / 2.0 + 0.5) / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def yolo_to_box(
    yolo: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Inverse of `box_to_yolo`, for tests and preview rendering."""
    cx, cy, w, h = yolo
    cx_px, cy_px = cx * width - 0.5, cy * height - 0.5
    w_px, h_px = w * width, h * height
    return (cx_px - w_px / 2, cy_px - h_px / 2, cx_px + w_px / 2, cy_px + h_px / 2)


def polygon_area(poly: np.ndarray) -> float:
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def visible_fraction(hull_img: np.ndarray, width: int, height: int) -> float:
    """Fraction of the projected hull's area that falls inside the image."""
    area = polygon_area(hull_img)
    if area <= 1e-9:
        return 0.0
    rect = np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]],
        dtype=np.float32,
    )
    inter, _ = cv2.intersectConvexConvex(hull_img.astype(np.float32), rect)
    return float(min(1.0, inter / area))


def hull_overlap(hull_a: np.ndarray, hull_b: np.ndarray) -> float:
    """Intersection area over the smaller hull's area."""
    inter, _ = cv2.intersectConvexConvex(hull_a.astype(np.float32), hull_b.astype(np.float32))
    if inter <= 0.0:
        return 0.0
    smaller = min(polygon_area(hull_a), polygon_area(hull_b))
    return float(inter / smaller) if smaller > 1e-9 else 1.0


def local_scale(h: np.ndarray, point: np.ndarray) -> float:
    """Smallest singular value of the projection's Jacobian at an asset-space point.

    Values below 1 mean the asset is being shrunk, which aliases unless it is pre-filtered.
    """
    p = np.array([point[0], point[1], 1.0])
    proj = h @ p
    w = proj[2]
    if abs(w) < 1e-12:
        return 1e-6
    q = proj[:2] / w
    j = (h[:2, :2] - np.outer(q, h[2, :2])) / w
    sv = np.linalg.svd(j, compute_uv=False)
    return float(max(sv.min(), 1e-6))


def pyramid_level(scale: float, max_level: int) -> int:
    """Pyramid level whose pixels are at least half the size of the destination's."""
    if scale >= 1.0:
        return 0
    return int(min(max_level, max(0, math.floor(math.log2(1.0 / scale)))))


def level_matrix(full_wh: tuple[int, int], level_wh: tuple[int, int]) -> np.ndarray:
    """Map level-k pixel coords to full-resolution coords (cv2.resize's centre alignment)."""
    sx = full_wh[0] / level_wh[0]
    sy = full_wh[1] / level_wh[1]
    return np.array(
        [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def hull_from_mask(mask: np.ndarray) -> np.ndarray:
    """Convex hull of a binary mask, in pixel-edge coordinates.

    Each hull vertex is expanded to its pixel's four corners before the final hull, so an
    identity projection reproduces the mask's exact extent.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("mask is empty: no arrow pixels found")
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    corners = np.vstack(
        [hull + np.array(d) for d in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
    )
    return cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2).astype(np.float64)


def hull_bbox(hull: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(hull[:, 0].min()),
        float(hull[:, 1].min()),
        float(hull[:, 0].max()),
        float(hull[:, 1].max()),
    )
