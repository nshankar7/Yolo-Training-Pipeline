"""Typed generation config: YAML files -> validated models.

Every model forbids unknown keys, so a typo in a config fails loudly instead of silently
leaving a default in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _ordered(v: tuple[float, float]) -> tuple[float, float]:
    if v[0] > v[1]:
        raise ValueError(f"range must be ordered [lo, hi], got {list(v)}")
    return v


Range = Annotated[tuple[float, float], AfterValidator(_ordered)]
IntRange = Annotated[tuple[int, int], AfterValidator(_ordered)]


def uniform(rng: np.random.Generator, r: tuple[float, float]) -> float:
    return float(rng.uniform(r[0], r[1]))


def uniform_int(rng: np.random.Generator, r: tuple[int, int]) -> int:
    return int(rng.integers(r[0], r[1] + 1))


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dist(Base):
    """A scalar distribution over a bounded range."""

    dist: Literal["uniform", "loguniform", "normal"] = "uniform"
    range: Range
    std: float | None = None

    def sample(self, rng: np.random.Generator) -> float:
        lo, hi = self.range
        if self.dist == "uniform":
            return float(rng.uniform(lo, hi))
        if self.dist == "loguniform":
            return float(np.exp(rng.uniform(np.log(max(lo, 1e-9)), np.log(max(hi, 1e-9)))))
        mean = (lo + hi) / 2.0
        std = self.std if self.std is not None else (hi - lo) / 6.0
        return float(np.clip(rng.normal(mean, std), lo, hi))


class Choice(Base):
    choices: list[int]
    weights: list[float]

    @model_validator(mode="after")
    def _check(self) -> Choice:
        if len(self.choices) != len(self.weights):
            raise ValueError("choices and weights must have the same length")
        if sum(self.weights) <= 0:
            raise ValueError("weights must sum to a positive number")
        return self

    def sample(self, rng: np.random.Generator) -> int:
        p = np.asarray(self.weights, dtype=np.float64)
        return int(rng.choice(self.choices, p=p / p.sum()))


class OutputCfg(Base):
    root: Path = Path("datasets/arrows_v1")
    image_size: tuple[int, int] = (960, 720)  # W, H — match the AUV camera's aspect
    jpeg_quality: IntRange = (70, 95)
    shard_size: int = 10000


class SplitCfg(Base):
    count: int = 0
    stream: int = 0
    start: int = 0  # first index; rounds advance this so images never repeat


class HoldoutCfg(Base):
    backgrounds: float = 0.1  # fraction of stock backgrounds reserved for val
    assets: bool = True  # hold out one variant per class (classes with >= 3 variants)


class SplitsCfg(Base):
    train: SplitCfg = SplitCfg(count=20000, stream=0)
    val: SplitCfg = SplitCfg(count=2000, stream=1)
    holdout: HoldoutCfg = HoldoutCfg()


class SceneCfg(Base):
    negative_prob: float = 0.10
    num_objects: Choice = Choice(choices=[1, 2, 3, 4], weights=[0.55, 0.25, 0.12, 0.08])
    camera_hfov_deg: Range = (60.0, 90.0)
    max_attempts_per_object: int = 20
    class_weights: dict[str, float] | Literal["uniform"] = "uniform"


class StockCfg(Base):
    dirs: list[Path] = Field(default_factory=lambda: [Path("assets/backgrounds/stock")])
    crop_area: Range = (0.3, 1.0)
    crop_ratio: Range = (0.8, 1.25)
    hflip_prob: float = 0.5
    regrade_prob: float = 0.8  # blend a land photo towards water colour
    regrade_distance_m: Range = (1.0, 6.0)


class ProceduralCfg(Base):
    gradient_power: Range = (0.5, 3.0)
    gradient_tilt: Range = (-0.3, 0.3)
    top_gain: Range = (1.0, 1.7)
    bottom_gain: Range = (0.15, 0.6)
    noise_octaves: IntRange = (3, 6)
    noise_amplitude: Range = (0.02, 0.12)


class BackgroundCfg(Base):
    source_probs: dict[str, float] = Field(
        default_factory=lambda: {"stock": 0.5, "procedural": 0.5}
    )
    stock: StockCfg = StockCfg()
    procedural: ProceduralCfg = ProceduralCfg()


class PoseCfg(Base):
    roll_std_deg: float = 6.0
    roll_clip_deg: float = 15.0
    yaw_deg: Range = (-60.0, 60.0)
    pitch_deg: Range = (-40.0, 40.0)
    max_view_angle_deg: float = 70.0


class PlacementCfg(Base):
    truncation_prob: float = 0.15
    inside_margin_frac: float = 0.02
    max_hull_overlap: float = 0.0
    max_roi_area_frac: float = 4.0  # reject absurdly magnified near-camera poses


class VisibilityCfg(Base):
    min_visible_fraction: float = 0.6
    min_side_px_train: float = 10.0
    # Contrast is measured before whole-frame effects, which only reduce it, so this
    # threshold carries margin for the noise and exposure applied afterwards.
    min_contrast_delta_e: float = 8.0
    min_transmission: float = 0.10  # how much of an object's light must survive the water


class AppearanceCfg(Base):
    hue_shift_deg: Range = (-8.0, 8.0)
    saturation: Range = (0.7, 1.2)
    brightness: Range = (0.6, 1.2)
    illumination_gradient: Range = (0.0, 0.4)
    edge_feather_px: Range = (0.0, 1.2)


class BoardCfg(Base):
    """The placard an arrow can be printed on.

    Generating the board (instead of baking one into the artwork) is deliberate: a fixed
    board colour would be the easiest feature in the image, and the model would learn the
    placard rather than the arrow.
    """

    prob: float = 0.5
    square: bool = True
    padding_frac: Range = (0.10, 0.40)
    corner_frac: Range = (0.0, 0.25)
    palette: list[tuple[int, int, int]] = Field(
        default_factory=lambda: [
            (47, 193, 211),   # teal, as in the supplied example
            (240, 240, 235),  # white
            (245, 200, 40),   # yellow
            (225, 110, 40),   # orange
            (35, 40, 50),     # near-black
        ]
    )
    colour_jitter: float = 0.12
    label_target: Literal["glyph", "board"] = "glyph"
    min_glyph_contrast_delta_e: float = 12.0  # the arrow must stand out from its own board


class ObjectCfg(Base):
    physical_size_m: Range = (0.3, 1.2)
    apparent_size: Dist = Dist(dist="loguniform", range=(0.03, 0.6))
    aspect_stretch: Dist = Dist(dist="loguniform", range=(0.87, 1.15))
    pose: PoseCfg = PoseCfg()
    placement: PlacementCfg = PlacementCfg()
    visibility: VisibilityCfg = VisibilityCfg()
    appearance: AppearanceCfg = AppearanceCfg()
    board: BoardCfg = BoardCfg()


class WaterType(Base):
    weight: float = 1.0
    beta: tuple[Range, Range, Range]  # per-metre attenuation ranges, R then G then B
    veil_rgb: tuple[IntRange, IntRange, IntRange]  # backscatter colour, sRGB 0-255


class UnderwaterCfg(Base):
    water_types: dict[str, WaterType]
    distance_clip_m: Range = (0.5, 15.0)
    forward_scatter_blur_k: Range = (0.0, 0.25)  # sigma_px = k * distance


class VignetteCfg(Base):
    prob: float = 0.5
    strength: Range = (0.1, 0.5)


class MotionBlurCfg(Base):
    prob: float = 0.25
    length_px: IntRange = (3, 15)


class DefocusCfg(Base):
    prob: float = 0.2
    sigma_px: Range = (0.5, 2.0)


class NoiseCfg(Base):
    prob: float = 0.8
    shot: Range = (0.0, 0.02)
    read: Range = (0.0, 0.015)


class ResolutionLossCfg(Base):
    prob: float = 0.2
    factor: Range = (0.4, 0.8)


class EffectsCfg(Base):
    vignetting: VignetteCfg = VignetteCfg()
    motion_blur: MotionBlurCfg = MotionBlurCfg()
    defocus: DefocusCfg = DefocusCfg()
    noise: NoiseCfg = NoiseCfg()
    resolution_loss: ResolutionLossCfg = ResolutionLossCfg()
    exposure_ev: Range = (-1.0, 0.7)
    wb_gain: Range = (0.85, 1.15)


class GenerateConfig(Base):
    version: int = 1
    seed: int = 1337
    classes_file: Path = Path("configs/classes.yaml")
    processed_dir: Path = Path("assets/arrows/processed")
    train_imgsz: int = 640
    output: OutputCfg = OutputCfg()
    splits: SplitsCfg = SplitsCfg()
    scene: SceneCfg = SceneCfg()
    background: BackgroundCfg = BackgroundCfg()
    object: ObjectCfg = ObjectCfg()
    underwater: UnderwaterCfg
    effects: EffectsCfg = EffectsCfg()

    def resolved(self, base_dir: Path) -> GenerateConfig:
        """Make every path absolute, relative to `base_dir`."""
        data = self.model_dump()

        def fix(p: Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (base_dir / p).resolve()

        data["classes_file"] = fix(data["classes_file"])
        data["processed_dir"] = fix(data["processed_dir"])
        data["output"]["root"] = fix(data["output"]["root"])
        data["background"]["stock"]["dirs"] = [fix(d) for d in data["background"]["stock"]["dirs"]]
        return GenerateConfig.model_validate(data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_sets(data: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    """Apply `--set a.b.c=value` overrides; the value is parsed as YAML."""
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {key}: {p} is not a mapping")
        node[parts[-1]] = value
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_generate_config(
    paths: list[Path], sets: list[str] | None = None, base_dir: Path | None = None
) -> GenerateConfig:
    """Load and deep-merge config files (later files win), then apply --set overrides."""
    data: dict[str, Any] = {}
    for p in paths:
        data = deep_merge(data, load_yaml(Path(p)))
    data = apply_sets(data, sets or [])
    cfg = GenerateConfig.model_validate(data)
    return cfg.resolved(Path(base_dir or Path.cwd()))


def config_hash(cfg: BaseModel) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
