"""Score trained weights on validation and stress sets.

Without real footage, the stress presets are the only way to see *where* a model is weak:
small and far, steep angles, low light, turbid water. Each is generated once (and reused
while its settings are unchanged), then scored with Ultralytics' own validator.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import load_generate_config
from .train import TrainConfig, _metrics_from
from .writer import generate_dataset, split_is_current, split_plans

STRESS_GLOB = "configs/presets/stress_*.yaml"


def stress_presets(pattern: str | None = None) -> list[Path]:
    return sorted(Path.cwd().glob(pattern or STRESS_GLOB))


def prepare_stress_sets(  # noqa: PLR0913
    base_configs: list[Path],
    sets: list[str] | None = None,
    presets: list[Path] | None = None,
    root: Path | None = None,
    workers: int = 0,
    progress: bool = True,
) -> dict[str, Path]:
    """Generate any stress set that is missing or stale. Returns {name: data.yaml}.

    Each preset is layered on top of `base_configs`, so a stress set shares the training
    config's camera, classes and artwork and differs only in the condition it stresses.
    `root`, when given, replaces each preset's output root with `root/<preset name>`.
    """
    out: dict[str, Path] = {}
    for preset in presets if presets is not None else stress_presets():
        name = preset.stem
        extra = list(sets or [])
        if root is not None:
            extra.append(f"output.root={Path(root) / name}")
        cfg = load_generate_config([*base_configs, preset], extra)
        plans = [p for p in split_plans(cfg) if p.count > 0]
        if not all(split_is_current(cfg, p) for p in plans):
            generate_dataset(cfg, workers=workers, overwrite=True, progress=progress)
        out[name] = Path(cfg.output.root) / "data.yaml"
    return out


def evaluate(
    weights: Path, datasets: dict[str, Path], train_cfg: TrainConfig
) -> list[dict[str, Any]]:
    """One row per dataset: overall metrics plus mAP50 for every class that appears."""
    from ultralytics import YOLO  # noqa: PLC0415 - heavy import, only needed here

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    for name, data_yaml in datasets.items():
        metrics = model.val(
            data=str(Path(data_yaml).resolve()),
            imgsz=train_cfg.imgsz,
            batch=train_cfg.batch,
            device=train_cfg.device,
            workers=train_cfg.workers,
            plots=False,
            project=str(Path(train_cfg.project).resolve()),
            name=f"eval_{name}",
            verbose=False,
        )
        row: dict[str, Any] = {"set": name, **_metrics_from(metrics)}
        for cls in metrics.summary():
            row[f"mAP50/{cls['Class']}"] = cls["mAP50"]
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    cols = ["set", "mAP50", "mAP50_95", "precision", "recall"]
    cols += sorted({k for r in rows for k in r if k.startswith("mAP50/")})

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return "-" if v is None else str(v)

    grid = [cols] + [[cell(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(line[i]) for line in grid) for i in range(len(cols))]
    return "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True)).rstrip()
        for line in grid
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
