"""Dataset statistics from the manifest.

The manifest records what was actually produced, including what was rejected and why. A high
rejection rate means the sampled ranges and the visibility thresholds disagree, and the
resulting distribution is not the one the config asks for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REJECTION_WARN_RATE = 0.4


def load_manifest(root: Path, split: str) -> list[dict]:
    path = Path(root) / "manifest" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no manifest for split '{split}': {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _histogram(values: list[float], bins: int, width: int = 34) -> list[str]:
    if not values:
        return ["  (none)"]
    arr = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:], strict=False):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  {lo:8.3f}-{hi:<8.3f} {c:6d} {bar}")
    return lines


def report(root: Path, split: str, bins: int = 12) -> str:
    metas = load_manifest(root, split)
    meta_doc_path = Path(root) / "dataset_meta.json"
    class_names: list[str] = []
    if meta_doc_path.exists():
        class_names = json.loads(meta_doc_path.read_text()).get("classes", [])

    counts: dict[str, int] = {n: 0 for n in class_names}
    sizes, distances, areas, aspects, deltas = [], [], [], [], []
    rejections: dict[str, int] = {}
    negatives = 0
    for m in metas:
        if not m["objects"]:
            negatives += 1
        for o in m["objects"]:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
            sizes.append(o["apparent_size"])
            distances.append(o["distance_m"])
            deltas.append(o["delta_e"])
            x0, y0, x1, y1 = o["box_px"]
            areas.append(float(np.sqrt(max(x1 - x0, 0) * max(y1 - y0, 0))))
            aspects.append((x1 - x0) / max(y1 - y0, 1e-6))
        for k, v in m["rejections"].items():
            rejections[k] = rejections.get(k, 0) + v

    total_objects = sum(counts.values())
    attempts = total_objects + sum(rejections.values())
    reject_rate = sum(rejections.values()) / max(attempts, 1)

    out = [
        f"split: {split}   images: {len(metas)}   objects: {total_objects}",
        f"background-only images: {negatives} ({negatives / max(len(metas), 1):.1%})",
        "",
        "class balance:",
    ]
    for name, c in sorted(counts.items()):
        share = c / max(total_objects, 1)
        out.append(f"  {name:<18} {c:7d}  {share:6.1%}  {'#' * int(round(40 * share))}")
    out += ["", "apparent size (long side / image long side):", *_histogram(sizes, bins)]
    out += ["", "box size (sqrt of area, px):", *_histogram(areas, bins)]
    out += ["", "distance (m):", *_histogram(distances, bins)]
    out += ["", "box aspect (w/h):", *_histogram(aspects, bins)]
    out += ["", "contrast against background (mean Lab delta-E):", *_histogram(deltas, bins)]
    out += ["", f"rejected placements: {sum(rejections.values())} of {attempts} attempts "
                f"({reject_rate:.1%})"]
    for k, v in sorted(rejections.items(), key=lambda kv: -kv[1]):
        out.append(f"  {k:<22} {v}")
    if reject_rate > REJECTION_WARN_RATE:
        out += [
            "",
            f"WARNING: over {REJECTION_WARN_RATE:.0%} of placements were rejected, so the "
            "produced distribution is skewed away from the configured ranges.",
            "Widen object.apparent_size, lower object.visibility thresholds, or reduce "
            "underwater attenuation.",
        ]
    return "\n".join(out)
