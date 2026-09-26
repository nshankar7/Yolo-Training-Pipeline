"""One image, start to finish: image = f(seed, stream, round, index).

The whole generator is this pure function. Nothing is stored between calls, so the same
index always yields the same image and labels, on any machine, at any worker count — and
the index space is unbounded, which is what makes the supply of training images endless.

Random draws are split into fixed per-stage streams, so changing (say) the effect settings
cannot shift where the objects land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backgrounds import make_background, water_colour
from .classes import ClassesConfig, load_classes
from .color import linear_to_srgb_u8
from .config import GenerateConfig, uniform, uniform_int
from .effects import apply_effects, sample_effects
from .geometry import (
    Pose,
    Projection,
    box_to_yolo,
    build_homography,
    camera_matrix,
    clip_box,
    distance_for_apparent_size,
    front_facing,
    hull_overlap,
    project_outline,
    visible_fraction,
)
from .registry import ROLE_TRAIN, ArrowAsset, AssetRegistry, BackgroundRegistry
from .render import (
    BoardSpec,
    ObjectAppearance,
    estimate_contrast,
    render_object,
    sample_appearance,
    sample_board,
)

N_STAGES = 6  # scene, background, objects, appearance, effects, encode

# The placement-time contrast estimate works from mean colours, so it is deliberately
# lenient: it only discards poses that are clearly hopeless, and the exact per-pixel check
# at render time has the final say.
CONTRAST_ESTIMATE_MARGIN = 0.6

# How many times a slot may re-place an object that failed its render-time checks.
RENDER_ATTEMPTS_PER_SLOT = 3


@dataclass
class Label:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Sample:
    image: np.ndarray  # uint8 BGR
    labels: list[Label]
    meta: dict
    jpeg_quality: int


@dataclass
class _Candidate:
    asset: ArrowAsset
    pose: Pose
    projection: Projection  # the sign's footprint: the board when there is one
    glyph_projection: Projection  # the arrow glyph alone
    board: BoardSpec | None
    appearance: ObjectAppearance
    apparent_size: float
    view_angle_deg: float
    visible: float

    def label_projection(self, label_target: str) -> Projection:
        """Which outline becomes the box: the whole placard, or just the arrow on it."""
        if self.board is not None and label_target == "board":
            return self.projection
        return self.glyph_projection


@dataclass
class GenContext:
    """Per-process state: config plus the asset and background inventories."""

    cfg: GenerateConfig
    classes: ClassesConfig
    assets: AssetRegistry
    backgrounds: BackgroundRegistry
    _class_weights: np.ndarray = field(default=None, repr=False)

    @classmethod
    def build(cls, cfg: GenerateConfig) -> GenContext:
        classes = load_classes(Path(cfg.classes_file))
        assets = AssetRegistry(cfg.processed_dir, classes, cfg.splits.holdout.assets)
        backgrounds = BackgroundRegistry(
            cfg.background.stock.dirs, cfg.splits.holdout.backgrounds
        )
        ctx = cls(cfg=cfg, classes=classes, assets=assets, backgrounds=backgrounds)
        ctx._class_weights = _class_weight_vector(cfg, classes)
        return ctx


def _class_weight_vector(cfg: GenerateConfig, classes: ClassesConfig) -> np.ndarray:
    names = classes.names
    if cfg.scene.class_weights == "uniform":
        return np.full(len(names), 1.0 / len(names))
    weights = np.zeros(len(names))
    for name, w in cfg.scene.class_weights.items():
        if name not in names:
            raise ValueError(f"scene.class_weights has unknown class {name!r}")
        weights[names.index(name)] = float(w)
    if weights.sum() <= 0:
        raise ValueError("scene.class_weights must contain at least one positive weight")
    return weights / weights.sum()


def _sample_water(
    rng: np.random.Generator, cfg: GenerateConfig
) -> tuple[str, np.ndarray, np.ndarray]:
    """Pick a water type, then its attenuation and backscatter colour (both linear BGR)."""
    types = sorted(cfg.underwater.water_types.items())
    weights = np.array([t.weight for _, t in types], dtype=np.float64)
    idx = int(rng.choice(len(types), p=weights / weights.sum()))
    name, spec = types[idx]
    beta_rgb = np.array([uniform(rng, r) for r in spec.beta], dtype=np.float32)
    beta_bgr = beta_rgb[::-1].copy()
    veil_lin_bgr = water_colour(rng, spec.veil_rgb)
    return name, beta_bgr, veil_lin_bgr


def _sample_pose_candidate(  # noqa: PLR0913 - placement genuinely depends on all of this
    rng: np.random.Generator,
    rng_appear: np.random.Generator,
    ctx: GenContext,
    k: np.ndarray,
    f: float,
    width: int,
    height: int,
    beta_bgr: np.ndarray,
    role: str,
    accepted: list[_Candidate],
    rejections: dict[str, int],
    canvas: np.ndarray,
    veil: np.ndarray,
) -> _Candidate | None:
    """Sample class, artwork, size and pose until one passes every check (or attempts run out)."""
    cfg = ctx.cfg
    obj = cfg.object
    vis, place, pose_cfg = obj.visibility, obj.placement, obj.pose
    long_side = max(width, height)

    for _ in range(cfg.scene.max_attempts_per_object):
        class_id = int(rng.choice(len(ctx.classes.names), p=ctx._class_weights))
        asset = ctx.assets.sample(rng, class_id, role)
        apparent = obj.apparent_size.sample(rng)
        stretch = obj.aspect_stretch.sample(rng)
        size_m = uniform(rng, obj.physical_size_m)

        # Distance follows from how big the sign is and how big it should look. In murky
        # water, a sign that far away would be invisible, so we bring it closer and let its
        # physical size shrink instead: the on-screen size distribution is what the detector
        # sees, so that is the one worth preserving.
        distance = distance_for_apparent_size(f, size_m, apparent, long_side)
        lo, hi = cfg.underwater.distance_clip_m
        beta_mean = float(np.mean(beta_bgr))
        visible_range = -math.log(max(vis.min_transmission, 1e-6)) / max(beta_mean, 1e-6)
        hi = min(hi, max(lo, visible_range))
        if distance < lo or distance > hi:
            distance = float(np.clip(distance, lo, hi))
            size_m = distance * apparent * long_side / f
            physical_lo, physical_hi = obj.physical_size_m
            if not (0.5 * physical_lo <= size_m <= 2.0 * physical_hi):
                rejections["implausible_size"] = rejections.get("implausible_size", 0) + 1
                continue

        transmission = float(np.exp(-beta_bgr * distance).mean())
        if transmission < vis.min_transmission:
            rejections["transmission"] = rejections.get("transmission", 0) + 1
            continue

        class_spec = ctx.classes.classes[class_id]
        # The global roll clip describes camera roll for designs that must stay upright. A
        # `uniform` class has declared its meaning survives rotation, so its own limit wins.
        if class_spec.roll_dist == "uniform":
            roll_clip = class_spec.max_roll_deg
            roll = float(rng.uniform(-roll_clip, roll_clip))
        else:
            roll_clip = min(pose_cfg.roll_clip_deg, class_spec.max_roll_deg)
            roll = float(np.clip(rng.normal(0.0, pose_cfg.roll_std_deg), -roll_clip, roll_clip))
        yaw = uniform(rng, pose_cfg.yaw_deg)
        pitch = uniform(rng, pose_cfg.pitch_deg)

        board = sample_board(rng, cfg.object.board, asset.hull_bbox)
        footprint = board.outline_asset if board is not None else asset.hull

        # Measure the sign's footprint with it centred, then pick a centre that fits.
        centre = ((width - 1) / 2.0, (height - 1) / 2.0)
        probe = Pose(roll, yaw, pitch, distance, centre, size_m, stretch)
        probe_proj = project_outline(build_homography(k, probe, asset.hull_bbox), footprint)
        if probe_proj.depths.min() <= 0:
            rejections["depth"] = rejections.get("depth", 0) + 1
            continue
        dx0, dy0, dx1, dy1 = (
            probe_proj.box[0] - centre[0], probe_proj.box[1] - centre[1],
            probe_proj.box[2] - centre[0], probe_proj.box[3] - centre[1],
        )
        margin_px = place.inside_margin_frac * min(width, height)
        truncate = rng.random() < place.truncation_prob
        if truncate:
            slack_x, slack_y = 0.4 * (dx1 - dx0), 0.4 * (dy1 - dy0)
            x_lo, x_hi = -0.5 - dx0 - slack_x, width - 0.5 - dx1 + slack_x
            y_lo, y_hi = -0.5 - dy0 - slack_y, height - 0.5 - dy1 + slack_y
        else:
            x_lo, x_hi = -0.5 + margin_px - dx0, width - 0.5 - margin_px - dx1
            y_lo, y_hi = -0.5 + margin_px - dy0, height - 0.5 - margin_px - dy1
        if x_hi < x_lo or y_hi < y_lo:
            rejections["no_placement_window"] = rejections.get("no_placement_window", 0) + 1
            continue

        pose = Pose(
            roll, yaw, pitch, distance,
            (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))),
            size_m, stretch,
        )
        facing, view_angle = front_facing(k, pose)
        if not facing:
            rejections["back_facing"] = rejections.get("back_facing", 0) + 1
            continue
        if view_angle > pose_cfg.max_view_angle_deg:
            rejections["view_angle"] = rejections.get("view_angle", 0) + 1
            continue

        homography = build_homography(k, pose, asset.hull_bbox)
        projection = project_outline(homography, footprint)
        glyph_projection = (
            project_outline(homography, asset.hull) if board is not None else projection
        )
        if board is not None:
            board.polygon_img = projection.hull_img
        if projection.depths.min() <= 0.05:
            rejections["depth"] = rejections.get("depth", 0) + 1
            continue

        label_target = cfg.object.board.label_target
        label_proj = (
            projection if (board is not None and label_target == "board") else glyph_projection
        )
        # Visibility is judged on whatever gets the label. Judging it on the board instead
        # would let an arrow sit almost entirely outside the frame and still be labelled,
        # while its larger board stayed 60% visible.
        seen = visible_fraction(label_proj.hull_img, width, height)
        if seen < vis.min_visible_fraction:
            rejections["visible_fraction"] = rejections.get("visible_fraction", 0) + 1
            continue

        box = clip_box(projection.box, width, height)
        label_box = clip_box(label_proj.box, width, height)
        if box is None or label_box is None:
            rejections["outside_image"] = rejections.get("outside_image", 0) + 1
            continue
        scale_to_train = cfg.train_imgsz / long_side
        if (
            min(label_box[2] - label_box[0], label_box[3] - label_box[1]) * scale_to_train
            < vis.min_side_px_train
        ):
            rejections["min_size"] = rejections.get("min_size", 0) + 1
            continue

        if any(
            hull_overlap(projection.hull_img, c.projection.hull_img) > place.max_hull_overlap
            for c in accepted
        ):
            rejections["overlap"] = rejections.get("overlap", 0) + 1
            continue

        appearance = sample_appearance(rng_appear, obj.appearance)
        estimated = estimate_contrast(
            canvas, veil, box, asset, appearance, beta_bgr, distance, board
        )
        if estimated < CONTRAST_ESTIMATE_MARGIN * vis.min_contrast_delta_e:
            rejections["low_contrast_estimate"] = rejections.get("low_contrast_estimate", 0) + 1
            continue

        return _Candidate(
            asset=asset,
            pose=pose,
            projection=projection,
            glyph_projection=glyph_projection,
            board=board,
            appearance=appearance,
            apparent_size=apparent,
            view_angle_deg=view_angle,
            visible=seen,
        )
    return None


def make_sample(ctx: GenContext, stream: int, index: int, role: str = ROLE_TRAIN,
                round_: int = 0) -> Sample:
    """Generate one image with its labels.

    Pure: depends only on (config, seed, stream, round, index).
    """
    cfg = ctx.cfg
    width, height = cfg.output.image_size
    seeds = np.random.SeedSequence([cfg.seed, stream, round_, index]).spawn(N_STAGES)
    rng_scene, rng_bg, rng_obj, rng_appear, rng_fx, rng_enc = (
        np.random.default_rng(s) for s in seeds
    )
    rejections: dict[str, int] = {}

    hfov = uniform(rng_scene, cfg.scene.camera_hfov_deg)
    k, f = camera_matrix(width, height, hfov)
    water_type, beta_bgr, veil_lin = _sample_water(rng_scene, cfg)
    negative = rng_scene.random() < cfg.scene.negative_prob
    n_objects = 0 if negative else cfg.scene.num_objects.sample(rng_scene)
    blur_k = uniform(rng_scene, cfg.underwater.forward_scatter_blur_k)

    # Background
    probs = cfg.background.source_probs
    want_stock = rng_bg.random() < probs.get("stock", 0.0) / max(
        probs.get("stock", 0.0) + probs.get("procedural", 0.0), 1e-9
    )
    bg_path = ctx.backgrounds.sample(rng_bg, role) if want_stock else None
    stock_img = ctx.backgrounds.load(bg_path) if bg_path is not None else None
    canvas, veil, bg_source = make_background(
        rng_bg, cfg.background, width, height, veil_lin, beta_bgr, stock_img
    )
    canvas = np.ascontiguousarray(canvas)

    # Place and draw one object at a time. Rendering can still reject a pose (an arrow that
    # turns out to be invisible against what it landed on), and drawing here means such a
    # slot can try again instead of silently becoming an empty image.
    #
    # Draw order is irrelevant while placement.max_hull_overlap is 0, because accepted hulls
    # never intersect. Allowing overlap would require drawing far-to-near and tracking each
    # object's remaining visible alpha.
    accepted: list[_Candidate] = []
    labels: list[Label] = []
    objects_meta: list[dict] = []

    for _ in range(n_objects):
        placed = False
        for _attempt in range(RENDER_ATTEMPTS_PER_SLOT):
            cand = _sample_pose_candidate(
                rng_obj, rng_appear, ctx, k, f, width, height, beta_bgr, role, accepted,
                rejections, canvas, veil,
            )
            if cand is None:
                break
            result = render_object(
                canvas=canvas,
                veil=veil,
                asset=cand.asset,
                projection=cand.projection,
                appearance=cand.appearance,
                beta_bgr=beta_bgr,
                distance_m=cand.pose.distance_m,
                blur_k=blur_k,
                visibility=cfg.object.visibility,
                max_roi_area_frac=cfg.object.placement.max_roi_area_frac,
                board=cand.board,
                glyph_contrast_min=cfg.object.board.min_glyph_contrast_delta_e,
            )
            if not result.ok:
                rejections[result.reason] = rejections.get(result.reason, 0) + 1
                continue
            placed = True
            break
        if not placed:
            rejections["slot_abandoned"] = rejections.get("slot_abandoned", 0) + 1
            continue

        accepted.append(cand)
        box = clip_box(
            cand.label_projection(cfg.object.board.label_target).box, width, height
        )
        if box is None:  # pragma: no cover - already checked during placement
            continue
        cx, cy, w, h = box_to_yolo(box, width, height)
        labels.append(Label(cand.asset.class_id, cx, cy, w, h))
        objects_meta.append(
            {
                "class": cand.asset.class_name,
                "class_id": cand.asset.class_id,
                "asset": cand.asset.key,
                "apparent_size": round(cand.apparent_size, 5),
                "distance_m": round(cand.pose.distance_m, 4),
                "size_m": round(cand.pose.size_m, 4),
                "stretch": round(cand.pose.stretch, 4),
                "roll_deg": round(cand.pose.roll_deg, 3),
                "yaw_deg": round(cand.pose.yaw_deg, 3),
                "pitch_deg": round(cand.pose.pitch_deg, 3),
                "view_angle_deg": round(cand.view_angle_deg, 3),
                "visible_fraction": round(cand.visible, 4),
                "board": cand.board is not None,
                "delta_e": round(result.delta_e, 3),
                "pyramid_level": result.level,
                "box_px": [round(v, 2) for v in box],
                "yolo": [round(v, 6) for v in (cx, cy, w, h)],
            }
        )

    effects = sample_effects(rng_fx, cfg.effects)
    canvas = apply_effects(canvas, effects, rng_fx)
    image = linear_to_srgb_u8(canvas)
    quality = uniform_int(rng_enc, cfg.output.jpeg_quality)

    meta = {
        "index": index,
        "stream": stream,
        "round": round_,
        "role": role,
        "image_size": [width, height],
        "hfov_deg": round(hfov, 3),
        "water_type": water_type,
        "beta_bgr": [round(float(b), 4) for b in beta_bgr],
        "forward_scatter_k": round(blur_k, 4),
        "background": {"source": bg_source, "path": str(bg_path) if bg_path else None},
        "label_target": cfg.object.board.label_target,
        "negative": bool(negative) or not labels,
        "n_objects_requested": n_objects,
        "objects": objects_meta,
        "rejections": rejections,
        "jpeg_quality": quality,
    }
    return Sample(image=image, labels=labels, meta=meta, jpeg_quality=quality)


def iter_samples(ctx: GenContext, stream: int, start: int = 0, role: str = ROLE_TRAIN):
    """An endless stream of samples — the 'infinite dataset' in its simplest form."""
    index = start
    while True:
        yield make_sample(ctx, stream, index, role=role)
        index += 1


def estimate_throughput(ctx: GenContext, n: int = 20) -> float:
    """Images per second, single process (used by `arrowgen bench`)."""
    import time  # noqa: PLC0415

    t0 = time.perf_counter()
    for i in range(n):
        make_sample(ctx, 0, i)
    return n / max(time.perf_counter() - t0, 1e-9)
