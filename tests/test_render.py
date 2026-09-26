"""Compositing, the water model, and the checks that decide whether an arrow is drawn."""

from __future__ import annotations

import numpy as np
import pytest
from test_geometry import HEIGHT, WIDTH, unit_scale_pose

from arrowgen.config import AppearanceCfg, BoardCfg, VisibilityCfg
from arrowgen.geometry import board_outline, build_homography, project_outline
from arrowgen.render import (
    BoardSpec,
    colour_matrix_bgr,
    render_object,
    sample_appearance,
    sample_board,
)

NO_JITTER = AppearanceCfg(
    hue_shift_deg=(0.0, 0.0), saturation=(1.0, 1.0), brightness=(1.0, 1.0),
    illumination_gradient=(0.0, 0.0), edge_feather_px=(0.0, 0.0),
)
LENIENT = VisibilityCfg(min_contrast_delta_e=0.0)
NO_WATER = np.zeros(3, np.float32)


def scene(colour=0.5):
    canvas = np.full((HEIGHT, WIDTH, 3), colour, np.float32)
    veil = np.full((HEIGHT, WIDTH, 3), 0.25, np.float32)
    return canvas, veil


def plain_appearance():
    return sample_appearance(np.random.default_rng(0), NO_JITTER)


def render(asset, canvas, veil, *, beta=NO_WATER, distance=1.0, visibility=LENIENT,
           board=None, blur_k=0.0, glyph_min=0.0, pose_kwargs=None):
    k, pose = unit_scale_pose(asset, **(pose_kwargs or {}))
    h = build_homography(k, pose, asset.hull_bbox)
    outline = board.outline_asset if board is not None else asset.hull
    projection = project_outline(h, outline)
    if board is not None:
        board.polygon_img = projection.hull_img
    return render_object(
        canvas=canvas, veil=veil, asset=asset, projection=projection,
        appearance=plain_appearance(), beta_bgr=beta, distance_m=distance, blur_k=blur_k,
        visibility=visibility, max_roi_area_frac=4.0, board=board, glyph_contrast_min=glyph_min,
    )


def test_a_successful_render_draws_the_glyph_and_leaves_the_rest_alone(rect_asset):
    canvas, veil = scene(0.8)
    before = canvas.copy()
    result = render(rect_asset, canvas, veil)
    assert result.ok
    changed = np.any(np.abs(canvas - before) > 1e-6, axis=2)
    assert changed.any()
    ys, xs = np.nonzero(changed)
    rx0, ry0, rx1, ry1 = result.roi
    assert rx0 <= xs.min() and xs.max() < rx1
    assert ry0 <= ys.min() and ys.max() < ry1


def test_an_invisible_arrow_is_not_drawn_at_all(rect_asset):
    """The invariant: no drawn pixels without a label, and no label without drawn pixels."""
    canvas, veil = scene(0.5)
    # Make the background exactly the glyph's own colour, so it cannot be seen.
    canvas[...] = rect_asset.mean_colour()[None, None, :]
    before = canvas.copy()
    result = render(rect_asset, canvas, veil, visibility=VisibilityCfg(min_contrast_delta_e=8.0))
    assert not result.ok
    assert result.reason == "low_contrast"
    assert np.array_equal(canvas, before), "a rejected object must leave no trace"


def test_compositing_leaves_no_dark_fringe(rect_asset):
    """Premultiplied alpha: edge pixels must stay between glyph and background, never darker."""
    canvas, veil = scene(1.0)  # white background
    result = render(rect_asset, canvas, veil, blur_k=2.0, distance=1.0)
    assert result.ok
    glyph = float(rect_asset.mean_colour().min())
    assert canvas.min() >= glyph - 0.02, "a fringe darker than both layers means a premultiply bug"
    assert canvas.max() <= 1.0 + 1e-6


def test_water_model_endpoints(rect_asset):
    """At zero distance the water does nothing; far away everything becomes the veil."""
    canvas, veil = scene(0.8)
    beta = np.array([0.5, 0.4, 0.3], np.float32)

    near_canvas = canvas.copy()
    render(rect_asset, near_canvas, veil, beta=beta, distance=0.0)
    no_water_canvas = canvas.copy()
    render(rect_asset, no_water_canvas, veil, beta=NO_WATER, distance=5.0)
    assert np.allclose(near_canvas, no_water_canvas, atol=1e-6)

    far_canvas = canvas.copy()
    result = render(rect_asset, far_canvas, veil, beta=beta, distance=400.0)
    assert result.ok
    centre = far_canvas[HEIGHT // 2, WIDTH // 2]
    assert np.allclose(centre, veil[HEIGHT // 2, WIDTH // 2], atol=1e-3)


def test_an_arrow_that_matches_its_board_is_refused(rect_asset):
    """A navy arrow on a navy placard is unreadable, however visible the placard is."""
    canvas, veil = scene(0.95)
    board = BoardSpec(
        colour_bgr=rect_asset.mean_colour().copy(),
        outline_asset=board_outline(rect_asset.hull_bbox, 0.25, True, 0.0),
    )
    before = canvas.copy()
    result = render(rect_asset, canvas, veil, board=board, glyph_min=12.0)
    assert not result.ok
    assert result.reason == "low_glyph_contrast"
    assert np.array_equal(canvas, before)


def test_a_contrasting_board_is_drawn_behind_the_glyph(rect_asset):
    canvas, veil = scene(0.95)
    board = BoardSpec(
        colour_bgr=np.array([0.05, 0.6, 0.6], np.float32),  # teal, unlike the navy glyph
        outline_asset=board_outline(rect_asset.hull_bbox, 0.3, True, 0.0),
    )
    result = render(rect_asset, canvas, veil, board=board, glyph_min=12.0)
    assert result.ok
    x0, y0, x1, y1 = (int(v) for v in project_outline(
        build_homography(*unit_scale_pose(rect_asset), rect_asset.hull_bbox),
        rect_asset.hull).box)
    # Just outside the glyph but inside the board, the board's colour should show.
    sample = canvas[y0 - 6, (x0 + x1) // 2]
    assert np.allclose(sample, board.colour_bgr, atol=0.05)


def test_board_sampling_respects_probability():
    bbox = (0.0, 0.0, 50.0, 50.0)
    rng = np.random.default_rng(3)
    assert sample_board(rng, BoardCfg(prob=0.0), bbox) is None
    assert sample_board(rng, BoardCfg(prob=1.0), bbox) is not None


def test_colour_matrix_is_neutral_by_default():
    assert np.allclose(colour_matrix_bgr(0.0, 1.0, 1.0), np.eye(3), atol=1e-6)
    assert colour_matrix_bgr(0.0, 1.0, 2.0)[0, 0] == pytest.approx(2.0)
