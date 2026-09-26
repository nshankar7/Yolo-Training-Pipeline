"""Asset and background inventories, with train/val holdouts.

Each worker process builds one of these once and reuses it, so decoded images and pyramids
are paid for once rather than per sample.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .classes import ClassesConfig
from .color import srgb_u8_to_linear
from .geometry import hull_bbox

MAX_PYRAMID_LEVELS = 6
ROLE_TRAIN = "train"
ROLE_VAL = "val"


def stable_fraction(text: str) -> float:
    """A deterministic value in [0, 1) derived from a string (holdout selection)."""
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


@dataclass
class ArrowAsset:
    class_id: int
    class_name: str
    path: Path
    hull: np.ndarray
    hull_bbox: tuple[float, float, float, float]
    size_wh: tuple[int, int]
    _pyramid: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)
    _mean_colour: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{self.class_name}/{self.path.stem}"

    def mean_colour(self) -> np.ndarray:
        """Alpha-weighted mean linear BGR of the glyph, for cheap contrast estimates."""
        if self._mean_colour is None:
            coarse, _ = self.pyramid()[-1]
            alpha = coarse[..., 3]
            weight = float(alpha.sum())
            rgb = coarse[..., :3].reshape(-1, 3).sum(axis=0)
            self._mean_colour = (rgb / max(weight, 1e-6)).astype(np.float32)
        return self._mean_colour

    def pyramid(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Premultiplied linear BGRA at halving resolutions, with each level's coord map.

        Premultiplied alpha is what makes downsampling, blurring and compositing correct;
        the pyramid is what stops distant arrows from aliasing into sparkle.
        """
        if self._pyramid:
            return self._pyramid
        from .geometry import level_matrix  # local import avoids a cycle at module load

        bgra = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape[2] != 4:
            raise ValueError(f"processed asset must be RGBA: {self.path}")
        lin = srgb_u8_to_linear(bgra[..., :3])
        alpha = (bgra[..., 3].astype(np.float32) / 255.0)[..., None]
        premul = np.dstack([lin * alpha, alpha]).astype(np.float32)
        full_wh = (premul.shape[1], premul.shape[0])
        levels = [(premul, np.eye(3))]
        cur = premul
        for _ in range(MAX_PYRAMID_LEVELS - 1):
            if min(cur.shape[:2]) <= 8:
                break
            cur = cv2.resize(
                cur, (max(1, cur.shape[1] // 2), max(1, cur.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            levels.append((cur, level_matrix(full_wh, (cur.shape[1], cur.shape[0]))))
        self._pyramid = levels
        return levels


class AssetRegistry:
    """All processed arrow assets, grouped by class."""

    def __init__(self, processed_dir: Path, classes_cfg: ClassesConfig, holdout_assets: bool):
        self.classes = classes_cfg
        self.by_class: dict[int, list[ArrowAsset]] = {}
        self.holdout: set[str] = set()
        processed_dir = Path(processed_dir)

        for class_id, name in enumerate(classes_cfg.names):
            class_dir = processed_dir / name
            paths = sorted(class_dir.glob("*.png"))
            if not paths:
                raise FileNotFoundError(
                    f"no processed assets for class '{name}' in {class_dir}. Run: arrowgen ingest"
                )
            assets = []
            for p in paths:
                meta_path = p.with_suffix(".json")
                if not meta_path.exists():
                    raise FileNotFoundError(
                        f"missing sidecar {meta_path.name}; re-run: arrowgen ingest"
                    )
                hull = np.array(json.loads(meta_path.read_text())["hull"], dtype=np.float64)
                size = json.loads(meta_path.read_text())["size"]
                assets.append(
                    ArrowAsset(
                        class_id=class_id,
                        class_name=name,
                        path=p,
                        hull=hull,
                        hull_bbox=hull_bbox(hull),
                        size_wh=(int(size[0]), int(size[1])),
                    )
                )
            self.by_class[class_id] = assets
        if holdout_assets:
            self._choose_holdout()

    def _choose_holdout(self) -> None:
        """Hold out one variant per class (classes with >= 3 variants) for validation.

        A derived class holds out the mirror of whatever its source held out. Picking it
        independently could put a design's mirror in val while the design itself trains
        under the source class, so val would not be unseen artwork.
        """
        names = self.classes.names
        held_stem: dict[str, str] = {}
        for class_id, assets in self.by_class.items():
            if self.classes.classes[class_id].derive is None and len(assets) >= 3:
                held = min(assets, key=lambda a: stable_fraction(a.key))
                held_stem[names[class_id]] = held.path.stem
                self.holdout.add(held.key)
        for class_id, assets in self.by_class.items():
            derive = self.classes.classes[class_id].derive
            if derive is None or len(assets) < 3:
                continue
            source_stem = held_stem.get(derive.source)
            paired = [a for a in assets if a.path.stem == f"{source_stem}__{derive.op}"]
            held = paired[0] if paired else min(assets, key=lambda a: stable_fraction(a.key))
            self.holdout.add(held.key)

    def assets_for(self, class_id: int, role: str) -> list[ArrowAsset]:
        assets = self.by_class[class_id]
        if not self.holdout:
            return assets
        if role == ROLE_VAL:
            held = [a for a in assets if a.key in self.holdout]
            return held or assets
        train = [a for a in assets if a.key not in self.holdout]
        return train or assets

    def sample(self, rng: np.random.Generator, class_id: int, role: str) -> ArrowAsset:
        pool = self.assets_for(class_id, role)
        return pool[int(rng.integers(len(pool)))]

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_class.values())


@lru_cache(maxsize=48)
def _load_background(path_str: str, max_side: int) -> np.ndarray:
    """Decode a stock background once per worker, capped in size to bound memory."""
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read background image: {path_str}")
    longest = max(img.shape[:2])
    if longest > max_side:
        s = max_side / longest
        img = cv2.resize(
            img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    return img


class BackgroundRegistry:
    """Stock background photos, split into train-only and val-only pools."""

    SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, dirs: list[Path], holdout_fraction: float, max_side: int = 2048):
        self.max_side = max_side
        paths: list[Path] = []
        for d in dirs:
            d = Path(d)
            if d.exists():
                paths += [p for p in sorted(d.rglob("*")) if p.suffix.lower() in self.SUFFIXES]
        self.paths = sorted(set(paths))
        self.val_only = {
            p for p in self.paths if stable_fraction(p.name) < max(0.0, holdout_fraction)
        }
        if self.paths and len(self.val_only) == len(self.paths):
            self.val_only = set()  # never leave the train pool empty

    def pool(self, role: str) -> list[Path]:
        if not self.paths:
            return []
        if role == ROLE_VAL:
            return sorted(self.val_only) or self.paths
        return [p for p in self.paths if p not in self.val_only] or self.paths

    def sample(self, rng: np.random.Generator, role: str) -> Path | None:
        pool = self.pool(role)
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    def load(self, path: Path) -> np.ndarray:
        return _load_background(str(path), self.max_side)
