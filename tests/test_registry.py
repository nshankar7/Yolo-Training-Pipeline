"""Asset registry: train/val holdouts."""

from __future__ import annotations

from arrowgen.registry import ROLE_TRAIN, ROLE_VAL, AssetRegistry


def test_a_derived_class_holds_out_the_mirror_of_its_source(project):
    """Otherwise val's loop_cw artwork could be the mirror of a loop_ccw training design."""
    registry = AssetRegistry(project["processed"], project["classes"], holdout_assets=True)
    held = {key.split("/")[0]: key.split("/")[1] for key in registry.holdout}
    assert set(held) == {"loop_ccw", "loop_cw"}
    assert held["loop_cw"] == f"{held['loop_ccw']}__hflip"


def test_held_out_assets_never_reach_training(project):
    registry = AssetRegistry(project["processed"], project["classes"], holdout_assets=True)
    for class_id in registry.by_class:
        train_keys = {a.key for a in registry.assets_for(class_id, ROLE_TRAIN)}
        val_keys = {a.key for a in registry.assets_for(class_id, ROLE_VAL)}
        assert train_keys and val_keys
        assert train_keys.isdisjoint(val_keys)
        assert val_keys <= registry.holdout
