"""Write a YOLO dataset to disk.

Workers write their own files and hand back only a small metadata record, so images are
never pickled between processes. Indices are handed out in order, which keeps the manifest
ordered and makes any single image reproducible from its index alone.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import __version__
from .config import GenerateConfig, config_hash
from .registry import ROLE_TRAIN, ROLE_VAL
from .sample import GenContext, make_sample

_WORKER_CTX: GenContext | None = None


@dataclass(frozen=True)
class SplitPlan:
    name: str
    stream: int
    start: int
    count: int
    role: str

    @property
    def indices(self) -> range:
        return range(self.start, self.start + self.count)


def split_plans(cfg: GenerateConfig) -> list[SplitPlan]:
    s = cfg.splits
    return [
        SplitPlan("train", s.train.stream, s.train.start, s.train.count, ROLE_TRAIN),
        SplitPlan("val", s.val.stream, s.val.start, s.val.count, ROLE_VAL),
    ]


# Measured: about 0.15 MB of JPEG per 960x720 image across the default quality range.
JPEG_BYTES_PER_PIXEL = 0.15e6 / (960 * 720)
DISK_HEADROOM_BYTES = 1 << 30  # leave 1 GB free for Ultralytics caches, runs and the OS


def estimate_bytes(n_images: int, image_size: tuple[int, int]) -> int:
    return int(n_images * image_size[0] * image_size[1] * JPEG_BYTES_PER_PIXEL)


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_disk(root: Path, needed: int, reclaimable: int = 0) -> None:
    """Fail before generating, not halfway through, when the images will not fit."""
    probe = Path(root)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free + reclaimable
    if needed + DISK_HEADROOM_BYTES > free:
        raise ValueError(
            f"not enough disk space under {probe}: need about {needed / 1e9:.1f} GB plus "
            f"{DISK_HEADROOM_BYTES / 1e9:.1f} GB headroom, but only {free / 1e9:.1f} GB is free. "
            "Generate fewer images, a smaller image_size, or free some space."
        )


def split_signature(cfg: GenerateConfig, plan: SplitPlan, round_: int) -> str:
    """Hash of everything that determines a split's images, and nothing else.

    Split counts and the output root are left out: they decide how many images exist and
    where they live, not what any one image looks like.
    """
    data = json.loads(cfg.model_dump_json())
    data.pop("splits")
    data["output"].pop("root")
    data["plan"] = [plan.name, plan.stream, plan.start, plan.count, plan.role, round_]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def split_is_current(cfg: GenerateConfig, plan: SplitPlan, round_: int = 0) -> bool:
    """True when this split already exists on disk, generated from the same settings."""
    meta = Path(cfg.output.root) / "manifest" / f"{plan.name}.meta.json"
    if not meta.exists():
        return False
    return json.loads(meta.read_text()).get("signature") == split_signature(cfg, plan, round_)


def image_path(root: Path, split: str, index: int, shard_size: int) -> Path:
    shard = index // max(shard_size, 1)
    return root / "images" / split / f"{shard:04d}" / f"{split}_{index:08d}.jpg"


def label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    # Ultralytics derives label paths by swapping the last '/images/' for '/labels/'.
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def _write_sample(plan: SplitPlan, index: int, round_: int) -> dict:
    assert _WORKER_CTX is not None, "worker context not initialised"
    cfg = _WORKER_CTX.cfg
    sample = make_sample(_WORKER_CTX, plan.stream, index, role=plan.role, round_=round_)
    img_p = image_path(Path(cfg.output.root), plan.name, index, cfg.output.shard_size)
    lbl_p = label_path(img_p)
    img_p.parent.mkdir(parents=True, exist_ok=True)
    lbl_p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(
        ".jpg", sample.image, [int(cv2.IMWRITE_JPEG_QUALITY), sample.jpeg_quality]
    )
    if not ok:  # pragma: no cover - only on a broken OpenCV build
        raise RuntimeError(f"failed to encode {img_p}")
    img_p.write_bytes(buf.tobytes())
    lbl_p.write_text("".join(f"{lab.to_line()}\n" for lab in sample.labels))
    meta = dict(sample.meta)
    meta["image"] = str(img_p.relative_to(cfg.output.root))
    meta["n_labels"] = len(sample.labels)
    return meta


def _init_worker(cfg_json: str) -> None:
    global _WORKER_CTX
    cv2.setNumThreads(1)  # one image per process; cv2's own threads would only fight for cores
    cfg = GenerateConfig.model_validate_json(cfg_json)
    _WORKER_CTX = GenContext.build(cfg)


def _task(args: tuple) -> dict:
    plan, index, round_ = args
    return _write_sample(plan, index, round_)


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git may be absent
        return None


def write_data_yaml(
    cfg: GenerateConfig,
    class_names: list[str],
    train_split: str = "train",
    val_split: str = "val",
    filename: str = "data.yaml",
) -> Path:
    """Ultralytics dataset descriptor.

    `path` is absolute on purpose: a relative path is resolved against Ultralytics' own
    datasets directory, not the current one, which silently points training somewhere else.
    """
    root = Path(cfg.output.root)
    lines = [
        "# Generated by arrowgen - do not edit by hand",
        f"path: {root}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(class_names)],
        "",
    ]
    out = root / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def summarise(metas: list[dict], class_names: list[str]) -> dict:
    counts = {n: 0 for n in class_names}
    rejections: dict[str, int] = {}
    sizes: list[float] = []
    distances: list[float] = []
    negatives = 0
    for m in metas:
        if not m["objects"]:
            negatives += 1
        for o in m["objects"]:
            counts[o["class"]] += 1
            sizes.append(o["apparent_size"])
            distances.append(o["distance_m"])
        for k, v in m["rejections"].items():
            rejections[k] = rejections.get(k, 0) + v
    total = sum(counts.values())
    return {
        "images": len(metas),
        "objects": total,
        "negatives": negatives,
        "negative_fraction": round(negatives / max(len(metas), 1), 4),
        "class_counts": counts,
        "class_share": {k: round(v / max(total, 1), 4) for k, v in counts.items()},
        "apparent_size": _quantiles(sizes),
        "distance_m": _quantiles(distances),
        "rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
    }


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values)
    q = np.quantile(arr, [0.05, 0.5, 0.95])
    return {"p05": round(float(q[0]), 4), "median": round(float(q[1]), 4),
            "p95": round(float(q[2]), 4), "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4)}


def generate_dataset(  # noqa: PLR0913
    cfg: GenerateConfig,
    workers: int = 0,
    round_: int = 0,
    overwrite: bool = False,
    progress: bool = True,
    plans: list[SplitPlan] | None = None,
    yaml_name: str = "data.yaml",
    check_space: bool = True,
) -> dict:
    """Generate the given splits and write data.yaml, the manifest and dataset_meta.json."""
    root = Path(cfg.output.root)
    plans = [p for p in (plans if plans is not None else split_plans(cfg)) if p.count > 0]
    if not plans:
        raise ValueError("nothing to generate: every split count is zero")

    occupied: list[Path] = []
    for plan in plans:
        targets = [
            root / "images" / plan.name,
            root / "labels" / plan.name,
            root / "manifest" / f"{plan.name}.jsonl",
            root / "manifest" / f"{plan.name}.meta.json",
        ]
        for target in targets:
            if not (target.is_file() or (target.is_dir() and any(target.rglob("*")))):
                continue
            if not overwrite:
                raise FileExistsError(
                    f"{target} already exists. Pass --overwrite to replace this split, "
                    "or choose a different output.root."
                )
            occupied.append(target)

    if check_space:
        needed = estimate_bytes(sum(p.count for p in plans), cfg.output.image_size)
        check_disk(root, needed, reclaimable=sum(_tree_bytes(t) for t in occupied))
    for target in occupied:
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    ctx = GenContext.build(cfg)
    class_names = ctx.classes.names
    cfg_json = cfg.model_dump_json()
    started = time.time()
    summary: dict = {"splits": {}}

    for plan in plans:
        (root / "manifest").mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest" / f"{plan.name}.jsonl"
        metas: list[dict] = []
        bar = tqdm(total=plan.count, desc=f"{plan.name}", disable=not progress, unit="img")
        with open(manifest, "w") as fh:
            if workers and workers > 1:
                mp_ctx = mp.get_context("spawn")
                args = [(plan, i, round_) for i in plan.indices]
                with mp_ctx.Pool(
                    processes=workers, initializer=_init_worker, initargs=(cfg_json,)
                ) as pool:
                    for meta in pool.imap(_task, args, chunksize=8):
                        fh.write(json.dumps(meta) + "\n")
                        metas.append(meta)
                        bar.update(1)
            else:
                global _WORKER_CTX
                _WORKER_CTX = ctx
                for i in plan.indices:
                    meta = _write_sample(plan, i, round_)
                    fh.write(json.dumps(meta) + "\n")
                    metas.append(meta)
                    bar.update(1)
        bar.close()
        summary["splits"][plan.name] = summarise(metas, class_names)
        # Written last, so a split interrupted mid-way never looks current.
        (root / "manifest" / f"{plan.name}.meta.json").write_text(json.dumps(
            {"signature": split_signature(cfg, plan, round_), "count": plan.count,
             "round": round_}
        ))

    val_name = next((p.name for p in plans if p.role == ROLE_VAL), "val")
    has_val_only = all(p.role == ROLE_VAL for p in plans)
    # A val-only set (a stress preset) must still name an existing train folder, or
    # Ultralytics refuses the whole data.yaml. Nothing trains on it.
    train_name = next(
        (p.name for p in plans if p.role == ROLE_TRAIN), val_name if has_val_only else "train"
    )
    data_yaml = write_data_yaml(cfg, class_names, train_name, val_name, yaml_name)
    meta_doc = {
        "arrowgen_version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - started, 2),
        "config_hash": config_hash(cfg),
        "config": json.loads(cfg.model_dump_json()),
        "classes": class_names,
        "assets": sorted(
            a.key for assets in ctx.assets.by_class.values() for a in assets
        ),
        "held_out_assets": sorted(ctx.assets.holdout),
        "stock_backgrounds": len(ctx.backgrounds.paths),
        "held_out_backgrounds": len(ctx.backgrounds.val_only),
        "round": round_,
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "summary": summary,
    }
    (root / "dataset_meta.json").write_text(json.dumps(meta_doc, indent=2))
    summary["data_yaml"] = str(data_yaml)
    summary["root"] = str(root)
    return summary
