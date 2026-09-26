"""Shared fixtures: a miniature but complete project (artwork, classes, config)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml

from arrowgen.classes import load_classes
from arrowgen.color import srgb_u8_to_linear
from arrowgen.config import GenerateConfig
from arrowgen.geometry import hull_bbox, hull_from_mask, level_matrix
from arrowgen.ingest import ingest_all
from arrowgen.placeholders import write_placeholder_assets
from arrowgen.registry import ArrowAsset
from arrowgen.sample import GenContext

CLASSES_YAML = {
    "classes": [
        {"name": "loop_ccw", "max_roll_deg": 180, "roll_dist": "uniform",
         "mirror": {"hflip": "loop_cw", "vflip": "loop_cw"}},
        {"name": "loop_cw", "max_roll_deg": 180, "roll_dist": "uniform",
         "mirror": {"hflip": "loop_ccw", "vflip": "loop_ccw"},
         "derive": {"from": "loop_ccw", "op": "hflip"}},
    ]
}

WATER = {
    "water_types": {
        "pool": {"weight": 0.5, "beta": [[0.15, 0.35], [0.03, 0.08], [0.02, 0.06]],
                 "veil_rgb": [[20, 90], [120, 200], [160, 235]]},
        "freshwater": {"weight": 0.5, "beta": [[0.5, 1.2], [0.3, 0.8], [0.6, 1.5]],
                       "veil_rgb": [[40, 115], [85, 150], [45, 95]]},
    }
}


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> dict:
    """A real ingest run over the placeholder artwork, shared by the whole session."""
    root = tmp_path_factory.mktemp("project")
    raw = root / "assets/arrows/raw"
    processed = root / "assets/arrows/processed"
    write_placeholder_assets(raw)
    classes_file = root / "classes.yaml"
    classes_file.write_text(yaml.safe_dump(CLASSES_YAML))
    classes_cfg = load_classes(classes_file)
    results = ingest_all(classes_cfg, raw, processed, max_side=256)
    return {
        "root": root,
        "raw": raw,
        "processed": processed,
        "classes_file": classes_file,
        "classes": classes_cfg,
        "ingested": results,
    }


def make_config(project: dict, out_dir, **overrides) -> GenerateConfig:
    data = {
        "seed": 7,
        "classes_file": str(project["classes_file"]),
        "processed_dir": str(project["processed"]),
        "output": {"root": str(out_dir), "image_size": [320, 240], "shard_size": 100},
        "splits": {"train": {"count": 4, "stream": 0}, "val": {"count": 2, "stream": 1}},
        "background": {"source_probs": {"stock": 0.0, "procedural": 1.0},
                       "stock": {"dirs": []}},
        "underwater": WATER,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return GenerateConfig.model_validate(data).resolved(out_dir)


@pytest.fixture
def ctx(project, tmp_path) -> GenContext:
    return GenContext.build(make_config(project, tmp_path / "ds"))


def synthetic_asset(mask: np.ndarray, colour=(40, 40, 220)) -> ArrowAsset:
    """Build an in-memory asset from a binary mask, bypassing files."""
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[..., :3] = np.array(colour, np.uint8)
    rgba[..., 3] = np.where(mask > 0, 255, 0)
    hull = hull_from_mask(rgba[..., 3] >= 128)
    asset = ArrowAsset(
        class_id=0, class_name="synthetic", path=None, hull=hull,
        hull_bbox=hull_bbox(hull), size_wh=(mask.shape[1], mask.shape[0]),
    )
    lin = srgb_u8_to_linear(rgba[..., :3])
    alpha = (rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    premul = np.dstack([lin * alpha, alpha]).astype(np.float32)
    levels = [(premul, np.eye(3))]
    cur = premul
    while min(cur.shape[:2]) > 8 and len(levels) < 5:
        cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)
        levels.append((cur, level_matrix((premul.shape[1], premul.shape[0]),
                                         (cur.shape[1], cur.shape[0]))))
    asset._pyramid = levels
    return asset


@pytest.fixture
def rect_asset() -> ArrowAsset:
    mask = np.zeros((80, 120), np.uint8)
    mask[10:70, 20:100] = 255  # an 80x60 rectangle
    return synthetic_asset(mask)


@pytest.fixture
def ell_asset() -> ArrowAsset:
    """An L shape: its convex hull is far smaller than its bounding box under rotation."""
    mask = np.zeros((120, 120), np.uint8)
    mask[20:100, 20:45] = 255
    mask[75:100, 20:100] = 255
    return synthetic_asset(mask)
