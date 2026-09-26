"""Label files: every value in range, every class id real, negatives genuinely empty."""

from __future__ import annotations

import numpy as np
import yaml
from conftest import make_config

from arrowgen.sample import GenContext, make_sample


def test_generated_labels_are_always_in_range(ctx):
    seen_classes = set()
    total = 0
    for index in range(40):
        sample = make_sample(ctx, stream=0, index=index)
        for label in sample.labels:
            total += 1
            seen_classes.add(label.class_id)
            assert 0 <= label.class_id < len(ctx.classes.names)
            assert np.isfinite([label.cx, label.cy, label.w, label.h]).all()
            assert label.w > 0 and label.h > 0
            assert label.cx - label.w / 2 >= -1e-6
            assert label.cy - label.h / 2 >= -1e-6
            assert label.cx + label.w / 2 <= 1 + 1e-6
            assert label.cy + label.h / 2 <= 1 + 1e-6
    assert total > 20
    assert seen_classes == set(range(len(ctx.classes.names))), "every class should appear"


def test_label_lines_are_parseable(ctx):
    sample = make_sample(ctx, stream=0, index=3)
    for label, line in zip(sample.labels, [lab.to_line() for lab in sample.labels], strict=True):
        parts = line.split()
        assert len(parts) == 5
        assert int(parts[0]) == label.class_id
        assert [float(p) for p in parts[1:]] == [
            round(v, 6) for v in (label.cx, label.cy, label.w, label.h)
        ]


def test_negative_images_carry_no_labels(project, tmp_path):
    cfg = make_config(project, tmp_path / "ds", scene={"negative_prob": 1.0})
    ctx = GenContext.build(cfg)
    for index in range(5):
        sample = make_sample(ctx, stream=0, index=index)
        assert sample.labels == []
        assert sample.meta["negative"] is True


def test_labelling_the_board_gives_a_bigger_box_than_the_glyph(project, tmp_path):
    common = {"object": {"board": {"prob": 1.0}}, "scene": {"negative_prob": 0.0}}
    glyph_ctx = GenContext.build(make_config(project, tmp_path / "g", **common))
    board_cfg = make_config(project, tmp_path / "b", **common)
    board_cfg.object.board.label_target = "board"
    board_ctx = GenContext.build(board_cfg)

    def by_pose(sample) -> dict:
        # The two configs share a seed but not every rejection, so the same slot can hold a
        # different object. Only the identical sign, in the identical pose, is comparable.
        return {
            (o["asset"], o["roll_deg"], o["yaw_deg"], o["pitch_deg"], o["distance_m"]): label
            for o, label in zip(sample.meta["objects"], sample.labels, strict=True)
        }

    compared = 0
    for index in range(30):
        glyphs = by_pose(make_sample(glyph_ctx, stream=0, index=index))
        boards = by_pose(make_sample(board_ctx, stream=0, index=index))
        for key in glyphs.keys() & boards.keys():
            glyph_label, board_label = glyphs[key], boards[key]
            if glyph_label.w < 0.9 and glyph_label.h < 0.9:  # ignore ones clipped by the frame
                assert board_label.w >= glyph_label.w
                assert board_label.h >= glyph_label.h
                compared += 1
    assert compared > 5


def test_apparent_size_matches_the_written_boxes(ctx):
    """A sanity check that the size knob means what it says on screen."""
    width, height = ctx.cfg.output.image_size
    long_side = max(width, height)
    for index in range(30):
        sample = make_sample(ctx, stream=0, index=index)
        for obj, label in zip(sample.meta["objects"], sample.labels, strict=True):
            drawn_long_side = max(label.w * width, label.h * height) / long_side
            # Perspective and clipping only ever shrink the on-screen size.
            assert drawn_long_side <= obj["apparent_size"] * 1.6 + 0.02


def _rolls(ctx, n: int = 40) -> list[float]:
    return [
        obj["roll_deg"]
        for index in range(n)
        for obj in make_sample(ctx, stream=0, index=index).meta["objects"]
    ]


def test_uniform_roll_classes_use_their_own_limit(ctx):
    """A loop arrow reads the same at any angle, so the camera-roll clip must not cap it."""
    rolls = _rolls(ctx)
    assert rolls
    assert max(abs(r) for r in rolls) > ctx.cfg.object.pose.roll_clip_deg
    assert all(abs(r) <= 180.0 for r in rolls)


def test_upright_classes_stay_inside_the_roll_clip(project, tmp_path):
    classes = yaml.safe_load(project["classes_file"].read_text())
    for spec in classes["classes"]:
        spec["roll_dist"] = "normal"
    upright = tmp_path / "upright.yaml"
    upright.write_text(yaml.safe_dump(classes))
    ctx = GenContext.build(make_config(project, tmp_path / "ds", classes_file=str(upright)))
    rolls = _rolls(ctx)
    assert rolls
    assert all(abs(r) <= ctx.cfg.object.pose.roll_clip_deg + 1e-6 for r in rolls)
