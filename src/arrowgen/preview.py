"""Contact sheets for eyeballing images and their labels.

Two modes, and the difference matters. `preview generate` draws boxes straight from the
generator; `preview --dataset` re-reads the written .jpg/.txt pairs, so it also proves the
files on disk say what the generator meant.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import yolo_to_box
from .sample import GenContext, Sample, make_sample

PALETTE = [
    (66, 135, 245), (60, 200, 120), (40, 190, 250), (200, 110, 240),
    (250, 160, 60), (90, 220, 220), (140, 120, 255), (70, 240, 170),
]


def class_colour(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def draw_labels(
    img: np.ndarray, labels, names: list[str], thickness: int = 2
) -> np.ndarray:
    """Draw normalised YOLO boxes (class_id, cx, cy, w, h) onto a copy of the image."""
    out = img.copy()
    h, w = out.shape[:2]
    for class_id, cx, cy, bw, bh in labels:
        x0, y0, x1, y1 = yolo_to_box((cx, cy, bw, bh), w, h)
        colour = class_colour(int(class_id))
        cv2.rectangle(out, (round(x0), round(y0)), (round(x1), round(y1)), colour, thickness)
        name = names[int(class_id)] if int(class_id) < len(names) else str(class_id)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(round(y0), th + 4)
        cv2.rectangle(out, (round(x0), ty - th - 4), (round(x0) + tw + 4, ty), colour, -1)
        cv2.putText(out, name, (round(x0) + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return out


def contact_sheet(images: list[np.ndarray], cols: int = 8, cell: int = 320) -> np.ndarray:
    if not images:
        raise ValueError("no images to lay out")
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    cell_h = round(cell * images[0].shape[0] / images[0].shape[1])
    sheet = np.full((rows * cell_h, cols * cell, 3), 30, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell, cell_h), interpolation=cv2.INTER_AREA)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell : (c + 1) * cell] = resized
    return sheet


def _labels_of(sample: Sample) -> list[tuple[float, ...]]:
    return [(lab.class_id, lab.cx, lab.cy, lab.w, lab.h) for lab in sample.labels]


def preview_generated(
    ctx: GenContext, n: int, stream: int, start: int, role: str, out_path: Path, cols: int = 8
) -> Path:
    images = []
    for i in range(start, start + n):
        sample = make_sample(ctx, stream, i, role=role)
        images.append(draw_labels(sample.image, _labels_of(sample), ctx.classes.names))
    sheet = contact_sheet(images, cols=cols)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def read_dataset_pairs(root: Path, split: str, limit: int) -> list[tuple[Path, Path]]:
    from .writer import label_path  # noqa: PLC0415 - avoids an import cycle

    img_dir = Path(root) / "images" / split
    imgs = sorted(p for p in img_dir.rglob("*.jpg"))[:limit]
    return [(p, label_path(p)) for p in imgs]


def preview_dataset(
    root: Path, split: str, n: int, names: list[str], out_path: Path, cols: int = 8
) -> Path:
    """Read written images and label files back, and draw what the files actually say."""
    images = []
    for img_p, lbl_p in read_dataset_pairs(root, split, n):
        img = cv2.imread(str(img_p))
        labels = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                if line.strip():
                    parts = line.split()
                    labels.append((int(parts[0]), *[float(v) for v in parts[1:5]]))
        images.append(draw_labels(img, labels, names))
    if not images:
        raise FileNotFoundError(f"no images found under {Path(root) / 'images' / split}")
    sheet = contact_sheet(images, cols=cols)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path
