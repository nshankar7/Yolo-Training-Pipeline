"""Ultralytics training wrapper, plus the rounds loop that consumes an endless dataset.

Two things here are not conveniences but correctness:

* Some augmentations would invalidate our labels. A horizontal flip turns one arrow design
  into its mirror, which is a *different class*; large rotations and perspective do the same
  for near-mirror designs. Ultralytics enables `fliplr=0.5` by default, so those keys are
  forced to zero and cannot be overridden from the command line.
* Each round writes a new train folder rather than replacing one. Ultralytics caches label
  scans per folder, and reusing a folder would train on a stale cache.
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import model_validator

from .config import Base, GenerateConfig, apply_sets, deep_merge, load_yaml
from .registry import ROLE_TRAIN, ROLE_VAL
from .writer import SplitPlan, check_disk, estimate_bytes, generate_dataset, split_is_current

FORCED_REASON = (
    "arrowgen forces this to 0: mirrored/rotated arrows belong to a different class, "
    "so this augmentation would silently mislabel the data"
)


class ForcedAugCfg(Base):
    """Augmentations that would change an arrow's class. Always zero."""

    fliplr: float = 0.0
    flipud: float = 0.0
    degrees: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    mixup: float = 0.0

    @model_validator(mode="after")
    def _all_zero(self) -> ForcedAugCfg:
        for key, value in self.model_dump().items():
            if value != 0.0:
                raise ValueError(f"forced.{key} must stay 0.0 - {FORCED_REASON}")
        return self


class AugCfg(Base):
    """Augmentations that are safe for this data."""

    mosaic: float = 1.0
    scale: float = 0.5
    translate: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    close_mosaic: int = 10


class RoundsCfg(Base):
    images_per_round: int = 20000
    epochs_per_round: int = 5
    keep_last: int = 2


class TrainConfig(Base):
    model: str = "yolo11n.pt"
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16
    workers: int = 4
    device: str = "cpu"  # works everywhere; set "mps" or a GPU index explicitly
    optimizer: str = "SGD"  # never "auto": it ignores lr0 and can change between rounds
    lr0: float = 0.01
    warmup_epochs: float = 3.0
    patience: int = 50
    seed: int = 0
    cache: bool = False
    val: bool = True
    plots: bool = True
    # Ultralytics accumulates gradients up to this nominal batch size before stepping the
    # optimizer. A round with fewer images than this never completes a step, so the weights
    # never change - see the warning in `train_rounds`.
    nbs: int = 64
    project: Path = Path("runs/arrowgen")
    name: str = "train"
    aug: AugCfg = AugCfg()
    forced: ForcedAugCfg = ForcedAugCfg()
    rounds: RoundsCfg = RoundsCfg()

    @model_validator(mode="after")
    def _check_optimizer(self) -> TrainConfig:
        if self.optimizer == "auto":
            raise ValueError(
                "optimizer must be explicit (e.g. SGD or AdamW): 'auto' ignores lr0 and may "
                "pick a different optimizer per round, making rounds incomparable"
            )
        return self


def load_train_config(paths: list[Path], sets: list[str] | None = None) -> TrainConfig:
    data: dict[str, Any] = {}
    for p in paths:
        data = deep_merge(data, load_yaml(Path(p)))
    return TrainConfig.model_validate(apply_sets(data, sets or []))


def build_args(cfg: TrainConfig, data_yaml: Path, **overrides: Any) -> dict:
    """Assemble Ultralytics arguments. Forced keys win over everything."""
    args: dict[str, Any] = {
        "data": str(Path(data_yaml).resolve()),
        "model": cfg.model,
        "imgsz": cfg.imgsz,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "workers": cfg.workers,
        "device": cfg.device,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "warmup_epochs": cfg.warmup_epochs,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "cache": cfg.cache,
        "val": cfg.val,
        "plots": cfg.plots,
        # Absolute, like data.yaml's `path`: Ultralytics resolves a relative project
        # directory against its own runs folder, not the current one.
        "project": str(Path(cfg.project).resolve()),
        "name": cfg.name,
        "nbs": cfg.nbs,
        **cfg.aug.model_dump(),
    }
    clashes = set(overrides) & set(cfg.forced.model_dump())
    if clashes:
        raise ValueError(f"cannot override {sorted(clashes)}: {FORCED_REASON}")
    args.update(overrides)
    args.update(cfg.forced.model_dump())  # last word, always
    return args


def _metrics_from(results: Any) -> dict[str, float | None]:
    d = getattr(results, "results_dict", None) or {}
    return {
        "mAP50": d.get("metrics/mAP50(B)"),
        "mAP50_95": d.get("metrics/mAP50-95(B)"),
        "precision": d.get("metrics/precision(B)"),
        "recall": d.get("metrics/recall(B)"),
    }


def run_train(cfg: TrainConfig, data_yaml: Path, **overrides: Any) -> tuple[Any, Path]:
    """One training run. Returns (results, run directory)."""
    from ultralytics import YOLO  # noqa: PLC0415 - heavy import, only needed here

    args = build_args(cfg, data_yaml, **overrides)
    model_name = args.pop("model")
    print(f"[arrowgen] training {model_name} on {args['data']}")
    forced = ', '.join(f'{k}={v}' for k, v in cfg.forced.model_dump().items())
    print(f"[arrowgen] label-safety: {forced}")
    model = YOLO(model_name)
    results = model.train(**args)
    save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
    return results, save_dir


def _append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_rounds(rounds_csv: Path) -> list[dict]:
    """Rows already written by earlier invocations, oldest first."""
    if not rounds_csv.exists():
        return []
    with open(rounds_csv, newline="") as fh:
        return list(csv.DictReader(fh))


def train_rounds(  # noqa: PLR0913
    gen_cfg: GenerateConfig,
    train_cfg: TrainConfig,
    rounds: int,
    workers: int = 0,
    progress: bool = True,
    keep_val: bool = True,
    fresh: bool = False,
) -> list[dict]:
    """Generate a fresh batch of images and train on it, repeatedly.

    Round r uses indices [start + r*N, start + (r+1)*N), so no image is ever seen twice.
    Validation data is generated once and held fixed, so rounds are comparable.

    `rounds` is the total wanted. Rounds already recorded in rounds.csv are skipped and
    training continues from the last one's weights, so an interrupted run can simply be
    started again. `fresh` sets the old history aside and starts from round 0.
    """
    root = Path(gen_cfg.output.root)
    rounds_csv = root / "rounds.csv"
    n = train_cfg.rounds.images_per_round

    if fresh and rounds_csv.exists():
        aside = rounds_csv.with_name(f"rounds.{time.strftime('%Y%m%d-%H%M%S')}.csv")
        rounds_csv.rename(aside)
        print(f"[arrowgen] previous history moved to {aside.name}")

    done = completed_rounds(rounds_csv)
    weights = train_cfg.model
    if done:
        last = done[-1]
        if int(last.get("images", n)) != n:
            raise ValueError(
                f"{rounds_csv} was written with images_per_round={last['images']}, not {n}; "
                "resuming would reuse or skip image indices. Pass --fresh to start over."
            )
        weights = last["weights"]
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"cannot resume: round {last['round']} weights are missing ({weights}). "
                "Pass --fresh to start over."
            )
    first = len(done)
    if first >= rounds:
        print(f"[arrowgen] {first} rounds already recorded in {rounds_csv}; nothing to do")
        return []
    if first:
        print(f"[arrowgen] resuming at round {first} from {weights}")

    epoch_images = n * train_cfg.rounds.epochs_per_round
    if epoch_images < train_cfg.nbs:
        print(
            f"[arrowgen] warning: {epoch_images} images per round is below nbs="
            f"{train_cfg.nbs}, so the optimizer may never step and the weights will not "
            f"change. Use more images per round, or set nbs to at most {max(1, n // 2)}."
        )

    val_plan = SplitPlan(
        "val", gen_cfg.splits.val.stream, gen_cfg.splits.val.start,
        gen_cfg.splits.val.count, ROLE_VAL,
    )
    reuse_val = keep_val and split_is_current(gen_cfg, val_plan)
    # Peak usage: the rounds kept on disk, the one being written, and val.
    peak = n * (train_cfg.rounds.keep_last + 1) + (0 if reuse_val else val_plan.count)
    check_disk(root, estimate_bytes(peak, gen_cfg.output.image_size))

    if not reuse_val:
        generate_dataset(
            gen_cfg, workers=workers, overwrite=True, progress=progress, plans=[val_plan],
            yaml_name="data_val_only.yaml", check_space=False,
        )

    # Each round is a fresh Ultralytics run, so the LR schedule restarts every round. The
    # final round closes mosaic for its second half; closing it for the whole round (as
    # close_mosaic >= epochs_per_round would) leaves no mosaic epochs at all.
    final_close_mosaic = min(train_cfg.aug.close_mosaic, train_cfg.rounds.epochs_per_round // 2)

    history: list[dict] = []
    for r in range(first, rounds):
        started = time.time()
        split_name = f"train_r{r:03d}"
        start_index = gen_cfg.splits.train.start + r * n
        plan = SplitPlan(split_name, gen_cfg.splits.train.stream, start_index, n, ROLE_TRAIN)
        generate_dataset(
            gen_cfg, workers=workers, round_=r, overwrite=True, progress=progress,
            plans=[plan], yaml_name=f"data_r{r:03d}.yaml", check_space=False,
        )
        data_yaml = root / f"data_r{r:03d}.yaml"

        final_round = r == rounds - 1
        results, save_dir = run_train(
            train_cfg,
            data_yaml,
            model=weights,
            epochs=train_cfg.rounds.epochs_per_round,
            name=f"{train_cfg.name}_r{r:03d}",
            warmup_epochs=train_cfg.warmup_epochs if r == 0 else 0.0,
            close_mosaic=final_close_mosaic if final_round else 0,
        )
        weights = str(save_dir / "weights" / "last.pt")
        row = {
            "round": r,
            "images": n,
            "train_index_start": start_index,
            "epochs": train_cfg.rounds.epochs_per_round,
            **_metrics_from(results),
            "weights": weights,
            "seconds": round(time.time() - started, 1),
        }
        _append_csv(rounds_csv, row)
        history.append(row)

        old = r - train_cfg.rounds.keep_last
        if old >= 0:
            for sub in ("images", "labels"):
                target = root / sub / f"train_r{old:03d}"
                if target.exists():
                    shutil.rmtree(target)
    return history
