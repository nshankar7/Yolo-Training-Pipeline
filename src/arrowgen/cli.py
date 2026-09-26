"""Command line: ingest -> generate -> preview/stats -> train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classes import load_classes
from .config import load_generate_config
from .ingest import ingest_all, review_sheet
from .placeholders import DESIGNS, write_placeholder_assets
from .registry import ROLE_TRAIN, ROLE_VAL

DEFAULT_CONFIG = Path("configs/generate.yaml")
DEFAULT_TRAIN_CONFIG = Path("configs/train.yaml")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c", "--config", action="append", type=Path, default=None,
        help="generation config; repeat to layer presets (later files win)",
    )
    p.add_argument(
        "--set", action="append", dest="sets", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set object.pose.yaw_deg='[-30, 30]'",
    )


def _load_cfg(args: argparse.Namespace):
    paths = args.config or [DEFAULT_CONFIG]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"config file(s) not found: {', '.join(missing)}")
    return load_generate_config(paths, args.sets)


def cmd_placeholders(args: argparse.Namespace) -> int:
    written = write_placeholder_assets(args.raw, tuple(args.design or ["loop"]))
    print(f"wrote {len(written)} placeholder source images under {args.raw}:")
    for p in written:
        print(f"  {p}")
    print("\nReplace these with the real artwork, then run: arrowgen ingest")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    classes_cfg = load_classes(args.classes)
    results = ingest_all(classes_cfg, args.raw, args.out, args.max_side)
    by_class: dict[str, int] = {}
    for r in results:
        by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
    print(f"processed {len(results)} assets into {args.out}")
    for name in classes_cfg.names:
        derived = classes_cfg.by_name(name).derive
        suffix = f"  (mirrored from {derived.source})" if derived else ""
        print(f"  [{classes_cfg.class_id(name)}] {name:<18} {by_class.get(name, 0)} assets{suffix}")
    sheet = review_sheet(results, args.review)
    print(f"\nreview sheet: {sheet}")
    print("Check that every cutout is clean and its red outline hugs the arrow.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .writer import generate_dataset  # noqa: PLC0415

    cfg = _load_cfg(args)
    if args.train is not None:
        cfg.splits.train.count = args.train
    if args.val is not None:
        cfg.splits.val.count = args.val
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out is not None:
        cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"root": args.out})})
        cfg = cfg.resolved(Path.cwd())

    summary = generate_dataset(
        cfg, workers=args.workers, overwrite=args.overwrite, progress=not args.quiet
    )
    print(f"\ndataset: {summary['root']}")
    print(f"data.yaml: {summary['data_yaml']}")
    for split, s in summary["splits"].items():
        print(
            f"  {split}: {s['images']} images, {s['objects']} objects, "
            f"{s['negatives']} background-only ({s['negative_fraction']:.1%})"
        )
        print(f"    class share: {json.dumps(s['class_share'])}")
        if s["rejections"]:
            top = list(s["rejections"].items())[:3]
            print(f"    top rejections: {', '.join(f'{k}={v}' for k, v in top)}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .preview import preview_dataset, preview_generated  # noqa: PLC0415
    from .sample import GenContext  # noqa: PLC0415

    cfg = _load_cfg(args)
    role = ROLE_VAL if args.split.startswith("val") else ROLE_TRAIN
    if args.dataset is not None:
        names = load_classes(cfg.classes_file).names
        out = preview_dataset(args.dataset, args.split, args.n, names, args.out, args.cols)
        print(f"wrote {out} (boxes read back from the written label files)")
        return 0
    ctx = GenContext.build(cfg)
    stream = cfg.splits.val.stream if role == ROLE_VAL else cfg.splits.train.stream
    out = preview_generated(ctx, args.n, stream, args.start, role, args.out, args.cols)
    print(f"wrote {out}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .stats import report  # noqa: PLC0415

    print(report(args.dataset, args.split))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .sample import GenContext, estimate_throughput  # noqa: PLC0415

    cfg = _load_cfg(args)
    ctx = GenContext.build(cfg)
    rate = estimate_throughput(ctx, args.n)
    width, height = cfg.output.image_size
    print(f"{rate:.1f} images/s single process ({width}x{height})")
    print(f"~{rate * args.workers:.0f} images/s estimated with {args.workers} workers")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import load_train_config, run_train, train_rounds  # noqa: PLC0415

    train_cfg = load_train_config(
        [args.train_config] if Path(args.train_config).exists() else [], args.set_train
    )
    if args.rounds and args.rounds > 1:
        cfg = _load_cfg(args)
        history = train_rounds(cfg, train_cfg, args.rounds, workers=args.workers)
        print(f"\ncompleted {len(history)} rounds; history: {Path(cfg.output.root) / 'rounds.csv'}")
        return 0

    data_yaml = args.data
    if data_yaml is None:
        cfg = _load_cfg(args)
        data_yaml = Path(cfg.output.root) / "data.yaml"
    if not Path(data_yaml).exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}. Run 'arrowgen generate' first.")
    _, save_dir = run_train(train_cfg, Path(data_yaml))
    print(f"\nrun directory: {save_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arrowgen",
        description="Synthetic turn-arrow dataset generator for Ultralytics YOLO",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("placeholders", help="write placeholder arrow artwork to get started")
    sp.add_argument("--raw", type=Path, default=Path("assets/arrows/raw"))
    sp.add_argument(
        "--design", action="append", choices=list(DESIGNS),
        help="which stand-in design(s) to write (default: loop)",
    )
    sp.set_defaults(func=cmd_placeholders)

    sp = sub.add_parser("ingest", help="turn raw artwork into RGBA cutouts with outlines")
    sp.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    sp.add_argument("--raw", type=Path, default=Path("assets/arrows/raw"))
    sp.add_argument("--out", type=Path, default=Path("assets/arrows/processed"))
    sp.add_argument("--max-side", type=int, default=768)
    sp.add_argument("--review", type=Path, default=Path("assets/arrows/processed/_review.jpg"))
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("generate", help="write a YOLO dataset")
    _add_config_args(sp)
    sp.add_argument("--train", type=int, help="number of training images")
    sp.add_argument("--val", type=int, help="number of validation images")
    sp.add_argument("--out", type=Path, help="output root (overrides config)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--workers", type=int, default=0, help="worker processes (0 or 1 = inline)")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("preview", help="contact sheet of images with their boxes")
    _add_config_args(sp)
    sp.add_argument("--dataset", type=Path, help="read a written dataset instead of generating")
    sp.add_argument("--split", default="train")
    sp.add_argument("-n", type=int, default=64)
    sp.add_argument("--start", type=int, default=0)
    sp.add_argument("--cols", type=int, default=8)
    sp.add_argument("--out", type=Path, default=Path("preview.jpg"))
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("stats", help="distribution report from a dataset manifest")
    sp.add_argument("--dataset", type=Path, required=True)
    sp.add_argument("--split", default="train")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("bench", help="measure generation throughput")
    _add_config_args(sp)
    sp.add_argument("-n", type=int, default=20)
    sp.add_argument("--workers", type=int, default=6)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("train", help="train YOLO on a generated dataset")
    _add_config_args(sp)
    sp.add_argument("-t", "--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    sp.add_argument("--data", type=Path, help="path to data.yaml")
    sp.add_argument("--rounds", type=int, default=0, help="train on N rounds of fresh images")
    sp.add_argument("--workers", type=int, default=0, help="generation workers between rounds")
    sp.add_argument("--set-train", action="append", default=[], metavar="KEY=VALUE")
    sp.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
