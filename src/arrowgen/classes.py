"""Class registry and mirror policy.

Class ids come from the explicit order in classes.yaml and are append-only, so retraining
against an extended class list never renumbers existing labels.

Mirroring is the reason this file exists. A design mirrored horizontally may *be* another
class ("turn left" becomes "turn right"), so flips are resolved here and applied once at
ingest. The generator itself never flips an asset unless a class maps onto itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .config import Base, load_yaml

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
NONE = "none"
SELF = "self"


class MirrorPolicy(Base):
    hflip: str = NONE  # "none" | "self" | <other class name>
    vflip: str = NONE


class DeriveSpec(Base):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(alias="from")
    op: Literal["hflip", "vflip", "rot180"] = "hflip"


class ClassSpec(Base):
    name: str
    max_roll_deg: float = 15.0
    # How in-plane rotation is drawn. A design whose meaning survives rotation (a loop arrow
    # still loops the same way upside down) can use the full range uniformly; a design that
    # reads as another class when turned far enough must stay near upright.
    roll_dist: Literal["normal", "uniform"] = "normal"
    mirror: MirrorPolicy = MirrorPolicy()
    derive: DeriveSpec | None = None

    @model_validator(mode="after")
    def _check_name(self) -> ClassSpec:
        if not NAME_RE.match(self.name):
            raise ValueError(f"invalid class name {self.name!r}")
        return self


class IngestOverride(Base):
    method: Literal["auto", "rgba", "colorkey", "photo"] = "auto"
    key_tolerance: float = 24.0  # Lab distance treated as "still background"
    # Keep background-coloured regions that the border cannot reach (for designs whose fill
    # happens to match their background). Off by default, so the hole in a ring-shaped
    # arrow is removed like any other background.
    keep_enclosed: bool = False


class ClassesConfig(Base):
    classes: list[ClassSpec]
    ingest_overrides: dict[str, IngestOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ClassesConfig:
        names = [c.name for c in self.classes]
        if not names:
            raise ValueError("classes.yaml must list at least one class")
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate class names: {sorted(dupes)}")
        known = set(names)

        for c in self.classes:
            for axis in ("hflip", "vflip"):
                target = getattr(c.mirror, axis)
                if target in (NONE, SELF):
                    continue
                if target not in known:
                    raise ValueError(f"{c.name}.mirror.{axis} -> unknown class {target!r}")
                back = getattr(self.by_name(target).mirror, axis)
                if back != c.name:
                    raise ValueError(
                        f"mirror map must be symmetric: {c.name}.{axis} -> {target}, "
                        f"but {target}.{axis} -> {back}"
                    )
            if c.derive is not None:
                if c.derive.source not in known:
                    raise ValueError(f"{c.name}.derive.from -> unknown class {c.derive.source!r}")
                if c.derive.source == c.name:
                    raise ValueError(f"{c.name}.derive.from cannot be itself")
                if self.by_name(c.derive.source).derive is not None:
                    raise ValueError(
                        f"{c.name}.derive.from -> {c.derive.source}, which is itself derived; "
                        "derive chains are not allowed"
                    )
                if c.derive.op in ("hflip", "vflip"):
                    mapped = getattr(self.by_name(c.derive.source).mirror, c.derive.op)
                    if mapped != c.name:
                        raise ValueError(
                            f"{c.name} derives from {c.derive.source} by {c.derive.op}, but "
                            f"{c.derive.source}.mirror.{c.derive.op} is {mapped!r}, not {c.name!r}"
                        )

        for name in self.ingest_overrides:
            if name not in known:
                raise ValueError(f"ingest_overrides has unknown class {name!r}")
        return self

    def by_name(self, name: str) -> ClassSpec:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def class_id(self, name: str) -> int:
        return self.names.index(name)

    def override_for(self, name: str) -> IngestOverride:
        return self.ingest_overrides.get(name, IngestOverride())


def load_classes(path: Path) -> ClassesConfig:
    return ClassesConfig.model_validate(load_yaml(Path(path)))
