"""Geometry is where labels come from, so these tests are the ones that matter most."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from arrowgen.geometry import (
    Pose,
    board_outline,
    box_to_yolo,
    build_homography,
    build_projection,
    camera_matrix,
    clip_box,
    front_facing,
    hull_from_mask,
    project_outline,
    rotation_matrix,
    visible_fraction,
    yolo_to_box,
)

WIDTH, HEIGHT = 640, 480
HFOV = 75.0


def unit_scale_pose(asset, **kwargs) -> tuple[np.ndarray, Pose]:
    """A pose whose projection maps one asset pixel to exactly one image pixel."""
    k, f = camera_matrix(WIDTH, HEIGHT, HFOV)
    x0, y0, x1, y1 = asset.hull_bbox
    longest = max(x1 - x0, y1 - y0)
    size_m = 1.0
    params = {
        "roll_deg": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0,
        "distance_m": f * size_m / longest,
        "center_px": ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0),
        "size_m": size_m, "stretch": 1.0,
    }
    params.update(kwargs)
    return k, Pose(**params)


def test_identity_projection_reproduces_the_glyph_extent(rect_asset):
    k, pose = unit_scale_pose(rect_asset)
    proj = build_projection(k, pose, rect_asset.hull, rect_asset.hull_bbox)
    x0, y0, x1, y1 = rect_asset.hull_bbox
    assert proj.box[2] - proj.box[0] == pytest.approx(x1 - x0, abs=1e-6)
    assert proj.box[3] - proj.box[1] == pytest.approx(y1 - y0, abs=1e-6)
    assert (proj.box[0] + proj.box[2]) / 2 == pytest.approx((WIDTH - 1) / 2.0, abs=1e-6)
    assert (proj.box[1] + proj.box[3]) / 2 == pytest.approx((HEIGHT - 1) / 2.0, abs=1e-6)


def test_translation_moves_the_box_by_the_same_amount(rect_asset):
    k, pose = unit_scale_pose(rect_asset)
    base = build_projection(k, pose, rect_asset.hull, rect_asset.hull_bbox)
    moved_pose = Pose(**{**pose.__dict__, "center_px": (pose.center_px[0] + 40.0,
                                                        pose.center_px[1] - 25.0)})
    moved = build_projection(k, moved_pose, rect_asset.hull, rect_asset.hull_bbox)
    assert moved.box[0] - base.box[0] == pytest.approx(40.0, abs=1e-6)
    assert moved.box[1] - base.box[1] == pytest.approx(-25.0, abs=1e-6)


def test_doubling_distance_halves_the_box(rect_asset):
    k, pose = unit_scale_pose(rect_asset)
    base = build_projection(k, pose, rect_asset.hull, rect_asset.hull_bbox)
    far = build_projection(
        k, Pose(**{**pose.__dict__, "distance_m": pose.distance_m * 2}),
        rect_asset.hull, rect_asset.hull_bbox,
    )
    assert (far.box[2] - far.box[0]) == pytest.approx((base.box[2] - base.box[0]) / 2, abs=1e-6)


def test_quarter_turn_swaps_width_and_height(rect_asset):
    k, pose = unit_scale_pose(rect_asset, roll_deg=90.0)
    proj = build_projection(k, pose, rect_asset.hull, rect_asset.hull_bbox)
    x0, y0, x1, y1 = rect_asset.hull_bbox
    assert proj.box[2] - proj.box[0] == pytest.approx(y1 - y0, abs=1e-6)
    assert proj.box[3] - proj.box[1] == pytest.approx(x1 - x0, abs=1e-6)


def _mask_box(asset, h, width=WIDTH, height=HEIGHT):
    src = asset.pyramid()[0][0][..., 3]
    warped = cv2.warpPerspective(src, h, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(warped > 0.5)
    if len(xs) < 20:
        return None
    return (xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5)


def test_box_matches_the_pixels_actually_drawn(ell_asset):
    """The real contract: no drawn pixel falls outside the label box."""
    k, _ = camera_matrix(WIDTH, HEIGHT, HFOV)
    rng = np.random.default_rng(1234)
    checked = 0
    for _ in range(200):
        pose = Pose(
            roll_deg=float(rng.uniform(-180, 180)), yaw_deg=float(rng.uniform(-55, 55)),
            pitch_deg=float(rng.uniform(-35, 35)), distance_m=float(rng.uniform(1.5, 8.0)),
            center_px=(float(rng.uniform(180, 460)), float(rng.uniform(140, 340))),
            size_m=float(rng.uniform(0.4, 1.0)), stretch=float(rng.uniform(0.9, 1.1)),
        )
        h = build_homography(k, pose, ell_asset.hull_bbox)
        box = _mask_box(ell_asset, h)
        if box is None:
            continue
        checked += 1
        predicted = project_outline(h, ell_asset.hull).box
        assert box[0] >= predicted[0] - 1.0
        assert box[1] >= predicted[1] - 1.0
        assert box[2] <= predicted[2] + 1.0
        assert box[3] <= predicted[3] + 1.0
    assert checked > 100


def test_hull_box_beats_corner_box_for_an_L_shape(ell_asset):
    """Using the asset's corners instead of its outline would inflate rotated boxes."""
    k, pose = unit_scale_pose(ell_asset, roll_deg=40.0)
    h = build_homography(k, pose, ell_asset.hull_bbox)
    hull_box = project_outline(h, ell_asset.hull).box
    w, hgt = ell_asset.size_wh
    corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, hgt - 0.5], [-0.5, hgt - 0.5]])
    corner_box = project_outline(h, corners).box
    hull_area = (hull_box[2] - hull_box[0]) * (hull_box[3] - hull_box[1])
    corner_area = (corner_box[2] - corner_box[0]) * (corner_box[3] - corner_box[1])
    assert hull_area < 0.85 * corner_area

    mask_box = _mask_box(ell_asset, h)
    assert mask_box[0] >= hull_box[0] - 1.0
    assert mask_box[2] <= hull_box[2] + 1.0


def test_back_facing_and_steep_poses_are_rejected(rect_asset):
    k, pose = unit_scale_pose(rect_asset)
    facing, angle = front_facing(k, pose)
    assert facing and angle < 1.0

    behind = Pose(**{**pose.__dict__, "yaw_deg": 150.0})
    facing, _ = front_facing(k, behind)
    assert not facing, "a sign seen from behind shows a mirrored arrow: it must be rejected"


def test_accepted_poses_never_mirror_the_glyph(rect_asset):
    """A mirrored projection would flip the design into its opposite class."""
    k, _ = camera_matrix(WIDTH, HEIGHT, HFOV)
    rng = np.random.default_rng(99)
    for _ in range(300):
        pose = Pose(
            roll_deg=float(rng.uniform(-180, 180)), yaw_deg=float(rng.uniform(-179, 179)),
            pitch_deg=float(rng.uniform(-179, 179)), distance_m=float(rng.uniform(1, 10)),
            center_px=(float(rng.uniform(0, WIDTH)), float(rng.uniform(0, HEIGHT))),
            size_m=0.6, stretch=1.0,
        )
        facing, angle = front_facing(k, pose)
        if not facing or angle > 70.0:
            continue
        h = build_homography(k, pose, rect_asset.hull_bbox)
        proj = project_outline(h, rect_asset.hull)
        assert proj.depths.min() > 0
        assert np.linalg.det(h) > 0, "accepted pose produced a mirrored projection"


def test_rotation_matrix_is_orthonormal():
    r = rotation_matrix(12.0, -34.0, 56.0)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_half_outside_object_is_clipped_and_half_visible(rect_asset):
    k, pose = unit_scale_pose(rect_asset)
    x0, _, x1, _ = rect_asset.hull_bbox
    half_width = (x1 - x0) / 2.0
    edge_pose = Pose(**{**pose.__dict__, "center_px": (-0.5 + 0.0, pose.center_px[1])})
    proj = build_projection(k, edge_pose, rect_asset.hull, rect_asset.hull_bbox)
    assert visible_fraction(proj.hull_img, WIDTH, HEIGHT) == pytest.approx(0.5, abs=0.02)

    clipped = clip_box(proj.box, WIDTH, HEIGHT)
    assert clipped[0] == pytest.approx(-0.5)
    assert clipped[2] == pytest.approx(half_width - 0.5, abs=1.0)
    cx, cy, w, h = box_to_yolo(clipped, WIDTH, HEIGHT)
    assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
    assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0


def test_yolo_conversion_round_trips():
    box = (12.25, 40.5, 180.75, 260.0)
    yolo = box_to_yolo(box, WIDTH, HEIGHT)
    back = yolo_to_box(yolo, WIDTH, HEIGHT)
    assert np.allclose(back, box, atol=1e-3)


def test_hull_from_mask_covers_every_lit_pixel():
    mask = np.zeros((40, 50), np.uint8)
    mask[8:30, 12:44] = 1
    hull = hull_from_mask(mask)
    assert hull[:, 0].min() == pytest.approx(11.5)
    assert hull[:, 0].max() == pytest.approx(43.5)
    assert hull[:, 1].min() == pytest.approx(7.5)
    assert hull[:, 1].max() == pytest.approx(29.5)


def test_board_outline_encloses_the_glyph():
    bbox = (10.0, 20.0, 60.0, 50.0)
    outline = board_outline(bbox, padding_frac=0.2, square=True, corner_frac=0.2)
    assert outline[:, 0].min() < bbox[0]
    assert outline[:, 1].min() < bbox[1]
    assert outline[:, 0].max() > bbox[2]
    assert outline[:, 1].max() > bbox[3]
    width = outline[:, 0].max() - outline[:, 0].min()
    height = outline[:, 1].max() - outline[:, 1].min()
    assert width == pytest.approx(height, rel=1e-6), "square board should be square"
