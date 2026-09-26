"""Turn raw artwork into clean RGBA cutouts with a stored outline.

Input can be a transparent PNG, flat art on a solid background, or a photo. Output is always
the same: a tight-cropped, padded RGBA PNG plus a JSON sidecar holding the convex hull of the
glyph. The generator projects that hull to get the box, so this is where label accuracy starts.

Mirrored classes are materialised here too: a class with a `derive` rule gets flipped copies
of its source class's assets. The generator then never has to flip anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .classes import ClassesConfig
from .geometry import hull_from_mask

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PAD_PX = 4
DEFAULT_MAX_SIDE = 768


@dataclass
class IngestedAsset:
    class_name: str
    source: Path | None
    path: Path
    method: str
    derived_from: str | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_image_bgra(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def has_usable_alpha(bgra: np.ndarray) -> bool:
    alpha = bgra[..., 3]
    return bool((alpha < 250).mean() > 0.01)


def alpha_from_colorkey(
    bgr: np.ndarray, tolerance: float, keep_enclosed: bool = False
) -> np.ndarray:
    """Alpha for flat art on a solid background.

    The background colour is read from the image border. By default every pixel of that
    colour is removed, including the hole inside a ring-shaped arrow. Set `keep_enclosed`
    when a design is *filled* with the background colour and only the outside should go.
    """
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    h, w = lab.shape[:2]
    border = np.concatenate(
        [lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3), lab[:, :2].reshape(-1, 3),
         lab[:, -2:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    ramp = np.clip((dist - 0.5 * tolerance) / (0.5 * tolerance + 1e-6), 0.0, 1.0).astype(
        np.float32
    )
    if not keep_enclosed:
        return ramp

    bg_like = (dist < tolerance).astype(np.uint8)
    _, labels = cv2.connectedComponents(bg_like, connectivity=4)
    border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)
    background = (
        np.isin(labels, list(border_labels))
        if border_labels
        else np.zeros_like(bg_like, bool)
    )

    alpha = np.ones((h, w), np.float32)
    alpha[background] = 0.0

    # Soften only the rim where glyph meets background, so anti-aliased edges survive.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    band = (cv2.dilate(background.astype(np.uint8), k) > 0) & (~background)
    alpha[band] = ramp[band]
    return alpha


def alpha_from_photo(bgr: np.ndarray) -> np.ndarray:
    try:
        from rembg import remove  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "photo ingest needs the 'photo' extra: uv sync --extra photo"
        ) from exc
    rgba = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
    return rgba[..., 3].astype(np.float32) / 255.0


def drop_specks(alpha: np.ndarray, min_area_frac: float = 0.01) -> np.ndarray:
    """Remove small disconnected blobs (JPEG noise, stray marks)."""
    solid = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 1:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {i + 1 for i, a in enumerate(areas) if a >= min_area_frac * areas.max()}
    keep_mask = np.isin(labels, list(keep)).astype(np.uint8)
    # Grow slightly so the anti-aliased rim of a kept blob survives with it.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return (alpha * cv2.dilate(keep_mask, k).astype(np.float32)).astype(np.float32)


def decontaminate(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Replace colours in partly transparent pixels with nearby opaque colours.

    Without this, an anti-aliased edge keeps a trace of its original background and shows
    up as a halo once the arrow is pasted somewhere else.
    """
    interior = (alpha > 0.9).astype(np.float32)
    if interior.sum() < 1:
        return bgr
    img = bgr.astype(np.float32)
    num = cv2.GaussianBlur(img * interior[..., None], (0, 0), 3.0)
    den = cv2.GaussianBlur(interior, (0, 0), 3.0)[..., None]
    filled = num / np.maximum(den, 1e-6)
    w = interior[..., None]
    return np.clip(img * w + filled * (1 - w), 0, 255).astype(np.uint8)


def tight_crop(bgra: np.ndarray, max_side: int) -> np.ndarray:
    ys, xs = np.nonzero(bgra[..., 3] > 0)
    if len(xs) == 0:
        raise ValueError("no opaque pixels after masking")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = bgra[y0:y1, x0:x1]
    longest = max(crop.shape[:2])
    if longest > max_side:
        s = max_side / longest
        crop = cv2.resize(
            crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.copyMakeBorder(
        crop, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )


def sidecar_path(png_path: Path) -> Path:
    return png_path.with_suffix(".json")


def write_asset(png_path: Path, bgra: np.ndarray, meta: dict) -> None:
    hull = hull_from_mask(bgra[..., 3] >= 128)
    cv2.imwrite(str(png_path), bgra)
    meta = {
        **meta,
        "size": [int(bgra.shape[1]), int(bgra.shape[0])],
        "hull": [[round(float(x), 3), round(float(y), 3)] for x, y in hull],
    }
    sidecar_path(png_path).write_text(json.dumps(meta, indent=2))


def process_file(  # noqa: PLR0913
    src: Path, class_name: str, method: str, tolerance: float, max_side: int,
    keep_enclosed: bool = False,
) -> tuple[np.ndarray, str, dict]:
    bgra = load_image_bgra(src)
    chosen = method
    if method == "auto":
        chosen = "rgba" if has_usable_alpha(bgra) else "colorkey"

    bgr = bgra[..., :3]
    if chosen == "rgba":
        alpha = bgra[..., 3].astype(np.float32) / 255.0
    elif chosen == "colorkey":
        alpha = alpha_from_colorkey(bgr, tolerance, keep_enclosed)
    elif chosen == "photo":
        alpha = alpha_from_photo(bgr)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown ingest method {chosen!r}")

    alpha = drop_specks(alpha)
    if (alpha > 0.5).sum() < 16:
        raise ValueError(
            f"{src.name}: almost nothing survived masking with method '{chosen}'. "
            "Try a different method in classes.yaml ingest_overrides."
        )
    bgr = decontaminate(bgr, alpha)
    bgra_out = np.dstack([bgr, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)])
    meta = {
        "class": class_name,
        "source": str(src),
        "sha256": _sha256(src),
        "method": chosen,
        "derived_from": None,
    }
    return tight_crop(bgra_out, max_side), chosen, meta


def apply_op(bgra: np.ndarray, op: str) -> np.ndarray:
    if op == "hflip":
        return np.ascontiguousarray(bgra[:, ::-1])
    if op == "vflip":
        return np.ascontiguousarray(bgra[::-1, :])
    if op == "rot180":
        return np.ascontiguousarray(bgra[::-1, ::-1])
    raise ValueError(f"unknown mirror op {op!r}")  # pragma: no cover


def _clear_processed(dir_: Path) -> None:
    if dir_.exists():
        for f in list(dir_.glob("*.png")) + list(dir_.glob("*.json")):
            f.unlink()


def ingest_all(
    classes_cfg: ClassesConfig,
    raw_dir: Path,
    processed_dir: Path,
    max_side: int = DEFAULT_MAX_SIDE,
) -> list[IngestedAsset]:
    """Process every raw asset, then materialise derived (mirrored) classes."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    known = set(classes_cfg.names)
    if raw_dir.exists():
        strays = sorted(d.name for d in raw_dir.iterdir() if d.is_dir() and d.name not in known)
        if strays:
            raise ValueError(
                f"raw folders with no matching class in classes.yaml: {strays}. "
                "Add the class or rename the folder."
            )

    results: list[IngestedAsset] = []
    for spec in classes_cfg.classes:
        out_dir = processed_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_processed(out_dir)
        if spec.derive is not None:
            continue
        override = classes_cfg.override_for(spec.name)
        src_dir = raw_dir / spec.name
        sources = sorted(
            p for p in src_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        ) if src_dir.exists() else []
        if not sources:
            raise ValueError(
                f"class '{spec.name}' has no artwork in {src_dir}. Add images, or give the "
                "class a 'derive' rule if it is the mirror of another class."
            )
        for src in sources:
            bgra, method, meta = process_file(
                src, spec.name, override.method, override.key_tolerance, max_side,
                override.keep_enclosed,
            )
            dst = out_dir / f"{src.stem}.png"
            write_asset(dst, bgra, meta)
            results.append(IngestedAsset(spec.name, src, dst, method))

    for spec in classes_cfg.classes:
        if spec.derive is None:
            continue
        out_dir = processed_dir / spec.name
        src_dir = processed_dir / spec.derive.source
        for src in sorted(src_dir.glob("*.png")):
            bgra = load_image_bgra(src)
            flipped = apply_op(bgra, spec.derive.op)
            dst = out_dir / f"{src.stem}__{spec.derive.op}.png"
            write_asset(
                dst,
                flipped,
                {
                    "class": spec.name,
                    "source": str(src),
                    "sha256": _sha256(src),
                    "method": f"derived:{spec.derive.op}",
                    "derived_from": spec.derive.source,
                },
            )
            results.append(
                IngestedAsset(spec.name, src, dst, f"derived:{spec.derive.op}", spec.derive.source)
            )

    for spec in classes_cfg.classes:
        if not list((processed_dir / spec.name).glob("*.png")):
            raise ValueError(f"class '{spec.name}' ended up with no processed assets")
    return results


def review_sheet(results: list[IngestedAsset], out_path: Path, cell: int = 220) -> Path:
    """Contact sheet of every processed asset on a checkerboard, with its hull outlined."""
    cols = min(6, max(1, len(results)))
    rows = (len(results) + cols - 1) // cols
    sheet = np.full((rows * (cell + 24), cols * cell, 3), 245, np.uint8)
    for i, res in enumerate(results):
        bgra = load_image_bgra(res.path)
        hull = np.array(json.loads(sidecar_path(res.path).read_text())["hull"], np.float32)
        s = cell / max(bgra.shape[:2])
        img = cv2.resize(bgra, (max(1, round(bgra.shape[1] * s)), max(1, round(bgra.shape[0] * s))))
        h, w = img.shape[:2]
        board = np.zeros((h, w, 3), np.uint8)
        tile = 12
        yy, xx = np.mgrid[0:h, 0:w]
        board[...] = np.where((((yy // tile) + (xx // tile)) % 2)[..., None] == 0, 200, 160)
        a = img[..., 3:4].astype(np.float32) / 255.0
        comp = (img[..., :3].astype(np.float32) * a + board.astype(np.float32) * (1 - a))
        comp = comp.astype(np.uint8).copy()
        cv2.polylines(comp, [np.round(hull * s).astype(np.int32)], True, (0, 0, 255), 1)
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + 24), c * cell
        sheet[y0 : y0 + h, x0 : x0 + w] = comp
        label = f"{res.class_name}/{res.path.stem}"[:34]
        cv2.putText(
            sheet, label, (x0 + 3, y0 + cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1,
            cv2.LINE_AA,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path
