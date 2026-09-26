"""Class registry rules — these prevent silently mislabelled mirror designs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arrowgen.classes import ClassesConfig


def build(classes: list[dict], **extra) -> ClassesConfig:
    return ClassesConfig.model_validate({"classes": classes, **extra})


def test_class_ids_follow_the_listed_order():
    cfg = build([{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert cfg.names == ["a", "b", "c"]
    assert [cfg.class_id(n) for n in cfg.names] == [0, 1, 2]


def test_appending_a_class_leaves_existing_ids_alone():
    before = build([{"name": "loop_ccw"}, {"name": "loop_cw"}])
    after = build([{"name": "loop_ccw"}, {"name": "loop_cw"}, {"name": "turn90_left"}])
    for name in before.names:
        assert after.class_id(name) == before.class_id(name)


def test_mirror_mapping_must_be_symmetric():
    with pytest.raises(ValidationError, match="symmetric"):
        build([
            {"name": "a", "mirror": {"hflip": "b"}},
            {"name": "b", "mirror": {"hflip": "none"}},
        ])


def test_mirror_target_must_exist():
    with pytest.raises(ValidationError, match="unknown class"):
        build([{"name": "a", "mirror": {"hflip": "nope"}}])


def test_self_mirror_is_allowed_for_symmetric_designs():
    cfg = build([{"name": "plus", "mirror": {"hflip": "self", "vflip": "self"}}])
    assert cfg.by_name("plus").mirror.hflip == "self"


def test_derive_must_agree_with_the_mirror_map():
    with pytest.raises(ValidationError, match="derives from"):
        build([
            {"name": "a", "mirror": {"hflip": "b"}},
            {"name": "b", "mirror": {"hflip": "a"}, "derive": {"from": "a", "op": "vflip"}},
        ])


def test_derive_chains_are_refused():
    with pytest.raises(ValidationError, match="itself derived"):
        build([
            {"name": "a", "mirror": {"hflip": "b"}},
            {"name": "b", "mirror": {"hflip": "a"}, "derive": {"from": "a", "op": "hflip"}},
            {"name": "c", "derive": {"from": "b", "op": "hflip"}},
        ])


def test_duplicate_names_are_refused():
    with pytest.raises(ValidationError, match="duplicate"):
        build([{"name": "a"}, {"name": "a"}])


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        build([{"name": "a", "maxroll": 12}])


def test_ingest_override_must_name_a_real_class():
    with pytest.raises(ValidationError, match="unknown class"):
        build([{"name": "a"}], ingest_overrides={"b": {"method": "rgba"}})


def test_roll_distribution_defaults_to_upright():
    cfg = build([{"name": "a"}, {"name": "b", "roll_dist": "uniform", "max_roll_deg": 180}])
    assert cfg.by_name("a").roll_dist == "normal"
    assert cfg.by_name("b").roll_dist == "uniform"
