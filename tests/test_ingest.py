"""Ingest: raw artwork in, clean cutouts plus outlines out."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml
from conftest import CLASSES_YAML

from arrowgen.classes import load_classes
from arrowgen.ingest import (
    alpha_from_colorkey,
    ingest_all,
    load_image_bgra,
    process_file,
    review_sheet,
    sidecar_path,
)
from arrowgen.placeholders import NAVY_BGR, TEAL_BGR, flat_on_background, loop_mask, rgba_cutout


def test_colour_key_removes_the_background_including_the_hole(tmp_path):
    """A ring-shaped arrow's centre is background, even though the border cannot reach it."""
    art = flat_on_background(loop_mask(256, 0.15), NAVY_BGR, TEAL_BGR)
    src = tmp_path / "flat.png"
    cv2.imwrite(str(src), art)

    bgra, method, _ = process_file(src, "loop_ccw", "auto", 24.0, 256)
    assert method == "colorkey"
    alpha = bgra[..., 3]
    centre = alpha[alpha.shape[0] // 2, alpha.shape[1] // 2]
    assert centre == 0, "the hole inside the ring should be transparent"
    assert (alpha > 200).mean() > 0.1, "the glyph itself must survive"
    kept = bgra[..., :3][alpha > 200]
    assert np.abs(kept.astype(int) - np.array(NAVY_BGR)).mean() < 40


def test_keep_enclosed_preserves_an_interior_fill():
    """A closed ring: its centre is unreachable from the border, so the mode decides."""
    ring = np.zeros((200, 200), np.uint8)
    cv2.circle(ring, (100, 100), 70, 255, 24)
    art = flat_on_background(ring, NAVY_BGR, TEAL_BGR)
    plain = alpha_from_colorkey(art, 24.0, keep_enclosed=False)
    enclosed = alpha_from_colorkey(art, 24.0, keep_enclosed=True)
    assert plain[100, 100] < 0.5 < enclosed[100, 100]


def test_rgba_artwork_keeps_its_own_alpha(tmp_path):
    art = rgba_cutout(loop_mask(200, 0.15), NAVY_BGR)
    src = tmp_path / "cutout.png"
    cv2.imwrite(str(src), art)
    bgra, method, _ = process_file(src, "loop_ccw", "auto", 24.0, 256)
    assert method == "rgba"
    assert bgra[..., 3].max() == 255


def test_ingest_derives_the_mirror_class(tmp_path):
    raw = tmp_path / "raw"
    (raw / "loop_ccw").mkdir(parents=True)
    cv2.imwrite(str(raw / "loop_ccw" / "a.png"),
                flat_on_background(loop_mask(200, 0.15), NAVY_BGR, TEAL_BGR))
    classes_file = tmp_path / "classes.yaml"
    classes_file.write_text(yaml.safe_dump(CLASSES_YAML))
    processed = tmp_path / "processed"

    ingest_all(load_classes(classes_file), raw, processed, max_side=256)
    source = load_image_bgra(processed / "loop_ccw" / "a.png")
    derived = load_image_bgra(processed / "loop_cw" / "a__hflip.png")
    assert np.array_equal(derived, source[:, ::-1])

    meta = json.loads(sidecar_path(processed / "loop_cw" / "a__hflip.png").read_text())
    assert meta["derived_from"] == "loop_ccw"
    assert meta["class"] == "loop_cw"


def test_hull_encloses_every_opaque_pixel(project):
    for res in project["ingested"]:
        bgra = load_image_bgra(res.path)
        hull = np.array(json.loads(sidecar_path(res.path).read_text())["hull"], np.float32)
        ys, xs = np.nonzero(bgra[..., 3] >= 128)
        assert xs.min() - 0.5 >= hull[:, 0].min() - 1e-6
        assert xs.max() + 0.5 <= hull[:, 0].max() + 1e-6
        assert ys.min() - 0.5 >= hull[:, 1].min() - 1e-6
        assert ys.max() + 0.5 <= hull[:, 1].max() + 1e-6


def test_raw_folder_with_no_matching_class_is_an_error(tmp_path):
    raw = tmp_path / "raw"
    (raw / "loop_ccw").mkdir(parents=True)
    (raw / "typo_class").mkdir(parents=True)
    cv2.imwrite(str(raw / "loop_ccw" / "a.png"),
                flat_on_background(loop_mask(120, 0.15), NAVY_BGR, TEAL_BGR))
    classes_file = tmp_path / "classes.yaml"
    classes_file.write_text(yaml.safe_dump(CLASSES_YAML))
    with pytest.raises(ValueError, match="no matching class"):
        ingest_all(load_classes(classes_file), raw, tmp_path / "processed")


def test_class_without_artwork_is_an_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    classes_file = tmp_path / "classes.yaml"
    classes_file.write_text(yaml.safe_dump(CLASSES_YAML))
    with pytest.raises(ValueError, match="no artwork"):
        ingest_all(load_classes(classes_file), raw, tmp_path / "processed")


def test_review_sheet_is_written(project, tmp_path):
    out = review_sheet(project["ingested"], tmp_path / "review.jpg")
    assert out.exists()
    sheet = cv2.imread(str(out))
    assert sheet is not None and sheet.shape[0] > 50
