# Yolo-Training-Pipeline

Synthetic training data for an underwater turn-arrow detector. You supply the arrow artwork;
the pipeline composites it into unlimited underwater scenes and **writes the YOLO labels
itself**, so nothing is labelled by hand.

```
assets/arrows/raw/<class>/*  ──ingest──►  clean RGBA cutouts + stored outlines
                                               │
stock photos ─┐                                ▼
procedural   ─┴──► scenes ──► place, light, submerge ──► images/ + labels/ + data.yaml ──► YOLO
```

## Quickstart

```bash
uv sync --extra dev --extra train      # one-time setup (Python 3.11 venv in .venv/)

uv run arrowgen placeholders           # stand-in artwork, replace with your own
uv run arrowgen ingest                 # cut out the arrows, derive mirrored classes
uv run arrowgen generate --train 20000 --val 2000 --workers 6
uv run arrowgen preview --dataset datasets/arrows_v1 --split train -n 64
uv run arrowgen train                  # Ultralytics, with label-unsafe augmentation disabled
```

## Using your own arrows

1. Put each design in its own folder: `assets/arrows/raw/<class_name>/whatever.png`.
   Several files per folder is good — variation in thickness, colour and wear all help.
   Transparent PNGs, flat graphics on a solid background, and photos all work.
2. Name the classes in [configs/classes.yaml](configs/classes.yaml). If one design is the
   mirror of another, say so there and give the second a `derive` rule; ingest then builds
   its artwork by flipping the first.
3. Run `uv run arrowgen ingest` and **open `assets/arrows/processed/_review.jpg`**. Every
   cutout should be clean and its red outline should hug the arrow. If a background was not
   removed, set `ingest_overrides` for that class in classes.yaml.

Flat artwork (an arrow on a coloured square) is colour-keyed automatically, including the
hole in the middle of a ring-shaped arrow. If a design is deliberately *filled* with its own
background colour, set `keep_enclosed: true` for it.

## Why the labels can be trusted

The box is not guessed from the pasted pixels; it is the arrow's stored outline pushed
through the same projection that drew it. Five rules protect it:

- **Nothing is drawn without a label.** An arrow too faint, too small or too far outside the
  frame is not left unlabelled — it is never drawn. A rejected object leaves the image
  byte-for-byte untouched, and the slot tries again.
- **Signs seen from behind are rejected.** The back of a sign shows a mirrored arrow, which
  is a different class. Poses with a back-facing normal or a viewing angle over 70° are
  refused outright.
- **Boxes come from the outline, not the artwork's corners.** For a loop or L-shaped arrow,
  corners would inflate every rotated box.
- **The generator never flips an arrow.** Mirrored classes get their own artwork at ingest.
- **Label-unsafe augmentation is forced off in training** (`fliplr`, `flipud`, `degrees`,
  `shear`, `perspective`, `mixup`). Ultralytics enables `fliplr=0.5` by default, which alone
  would mislabel about half the data.

Per class you can say how rotation behaves. A loop arrow still loops the same way upside
down (`roll_dist: uniform`, `max_roll_deg: 180`); a left-turn arrow does not, and must stay
near upright.

## Sign boards

Arrows are often printed on a placard. The pipeline **generates** that board rather than
baking it into the artwork, so its colour, padding and corner rounding vary, and half the
time there is no board at all. A fixed board colour would be the easiest feature in the
image and the model would learn the placard instead of the arrow.

With a board, two separate checks must pass: the sign must be visible against the scene, and
the arrow must be readable against its own board. `object.board.label_target` chooses whether
the box covers just the arrow (default) or the whole placard.

## Backgrounds

- **Stock photos**: drop any images into `assets/backgrounds/stock/`. Land photos are fine —
  they are regraded through the water model so they look submerged.
- **Procedural**: water scenes generated from the sampled water type, used when no stock
  photos are present. Two presets ship: clear `pool` water and murky `freshwater`.

Objects are attenuated by their own distance (`I = J·t + B·(1−t)`, `t = exp(−β·d)`), so a
far arrow is hazier and bluer than a near one, and haze colour is read from the scene.

## Commands

| Command | Purpose |
|---|---|
| `arrowgen placeholders` | Write stand-in artwork (`--design loop\|turn90\|uturn`) |
| `arrowgen ingest` | Cut out arrows, derive mirrors, write the review sheet |
| `arrowgen generate` | Write a YOLO dataset (`--train`, `--val`, `--workers`, `--overwrite`) |
| `arrowgen preview` | Contact sheet; `--dataset` reads the written files back |
| `arrowgen stats` | Class balance, size/distance histograms, rejection reasons |
| `arrowgen bench` | Generation throughput |
| `arrowgen train` | Train Ultralytics; `--rounds N` retrains on fresh images each round |

Configs layer: `-c configs/generate.yaml -c configs/presets/pool.yaml`, and single values
override with `--set object.pose.yaw_deg='[-30, 30]'`.

## Unlimited images

Every image is a pure function of `(config, seed, stream, round, index)`, so the index space
is unbounded and any image can be rebuilt exactly. `arrowgen train --rounds N` generates a
fresh batch each round from indices that have never been used, trains on it from the previous
weights, logs to `rounds.csv`, and prunes old rounds.

## Checking the data

```bash
uv run pytest                                          # 59 tests
uv run arrowgen preview --dataset datasets/arrows_v1 --split train -n 64
uv run arrowgen stats --dataset datasets/arrows_v1 --split train
```

The test suite checks that projected boxes match the pixels actually drawn, that accepted
poses never mirror a glyph, that an invisible arrow leaves no trace, that generation is
byte-identical whatever the worker count, and that derived mirror classes really are mirrors.

Also look at Ultralytics' own `labels.jpg` and `train_batch*.jpg` in the run directory: they
show what the model actually receives.

Without real footage, use the stress presets to find weaknesses per condition:

```bash
for p in stress_small_far stress_wide_angle stress_low_light stress_turbid; do
  uv run arrowgen generate -c configs/generate.yaml -c configs/presets/$p.yaml --overwrite
done
```

Add a real hand-labelled set later by pointing `--data` at it; no code change is needed.

## Hardware notes

- Generation is CPU-bound. Measured on an M2 with `--workers 6`: 29 images/s at 960×720
  (20k images ≈ 12 min, ≈ 3 GB of JPEGs) and 61 images/s at 640×480. Check free disk before
  generating a large set.
- Training on this Mac runs on CPU. MPS needs macOS 14+ with current torch; on macOS 13,
  `uv pip install torch==2.4.1 torchvision==0.19.1` restores MPS.
- For real training use a cloud NVIDIA GPU. Generate locally, upload a tar of the dataset,
  and keep the same pinned Ultralytics version and the same forced augmentation settings.
- Ultralytics is AGPL-3.0 — check that against how the vehicle software is distributed.
