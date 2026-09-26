"""Training orchestration: resumable rounds, fixed validation data, disk and config guards.

Ultralytics itself is replaced by a fake, so these run in seconds and need no weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_config

from arrowgen import train as train_mod
from arrowgen.cli import main
from arrowgen.train import TrainConfig, build_args, completed_rounds, train_rounds
from arrowgen.writer import check_disk, generate_dataset


@pytest.fixture
def fake_training(monkeypatch, tmp_path):
    """Record every run_train call and write a last.pt, as a real run would."""
    calls: list[dict] = []

    def fake_run_train(cfg, data_yaml, **overrides):
        # Like Ultralytics, never reuse a run directory: train_r000, train_r0002, ...
        save_dir, n = tmp_path / "runs" / overrides["name"], 2
        while save_dir.exists():
            save_dir, n = tmp_path / "runs" / f"{overrides['name']}{n}", n + 1
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "last.pt").write_bytes(b"weights")
        calls.append({"data": Path(data_yaml), **overrides})
        return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5}), save_dir

    monkeypatch.setattr(train_mod, "run_train", fake_run_train)
    return calls


def small_train_cfg(**rounds) -> TrainConfig:
    return TrainConfig.model_validate(
        {"nbs": 1, "rounds": {"images_per_round": 3, "epochs_per_round": 4, **rounds}}
    )


def test_rounds_resume_from_the_last_recorded_round(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=1, progress=False)
    assert len(fake_training) == 1

    history = train_rounds(cfg, small_train_cfg(), rounds=3, progress=False)
    assert [row["round"] for row in history] == [1, 2]
    assert fake_training[1]["model"].endswith("train_r000/weights/last.pt")
    assert fake_training[2]["model"].endswith("train_r001/weights/last.pt")
    assert [int(r["round"]) for r in completed_rounds(tmp_path / "ds" / "rounds.csv")] == [0, 1, 2]
    # Round r starts at index r * images_per_round, so no image is ever reused.
    assert [int(r["train_index_start"]) for r in completed_rounds(tmp_path / "ds" / "rounds.csv")] \
        == [0, 3, 6]


def test_a_finished_run_does_nothing_when_started_again(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=2, progress=False)
    assert train_rounds(cfg, small_train_cfg(), rounds=2, progress=False) == []
    assert len(fake_training) == 2


def test_fresh_sets_the_history_aside(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=1, progress=False)
    history = train_rounds(cfg, small_train_cfg(), rounds=1, progress=False, fresh=True)
    assert [row["round"] for row in history] == [0]
    assert fake_training[1]["model"] == TrainConfig().model
    assert list((tmp_path / "ds").glob("rounds.*.csv")), "old history should be kept aside"


def test_resume_refuses_missing_weights(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=1, progress=False)
    (tmp_path / "runs" / "train_r000" / "weights" / "last.pt").unlink()
    with pytest.raises(FileNotFoundError, match="--fresh"):
        train_rounds(cfg, small_train_cfg(), rounds=2, progress=False)


def test_resume_refuses_a_different_round_size(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=1, progress=False)
    with pytest.raises(ValueError, match="images_per_round"):
        train_rounds(cfg, small_train_cfg(images_per_round=5), rounds=2, progress=False)


def test_val_is_reused_only_when_its_settings_match(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=1, progress=False)
    val_meta = tmp_path / "ds" / "manifest" / "val.meta.json"
    stamp = val_meta.stat().st_mtime_ns

    train_rounds(cfg, small_train_cfg(), rounds=2, progress=False)
    assert val_meta.stat().st_mtime_ns == stamp, "unchanged settings should reuse val"

    changed = make_config(project, tmp_path / "ds", seed=99)
    train_rounds(changed, small_train_cfg(), rounds=3, progress=False)
    assert val_meta.stat().st_mtime_ns != stamp, "a new seed must regenerate val"


def test_final_round_keeps_some_mosaic_epochs(project, tmp_path, fake_training):
    cfg = make_config(project, tmp_path / "ds")
    train_rounds(cfg, small_train_cfg(), rounds=2, progress=False)
    assert fake_training[0]["close_mosaic"] == 0
    assert fake_training[1]["close_mosaic"] == 2  # min(10, 4 // 2)


def test_forced_augmentations_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError, match="fliplr"):
        build_args(TrainConfig(), tmp_path / "data.yaml", fliplr=0.5)
    assert build_args(TrainConfig(), tmp_path / "data.yaml")["fliplr"] == 0.0


def test_disk_guard_stops_before_writing_anything(project, tmp_path):
    with pytest.raises(ValueError, match="not enough disk space"):
        check_disk(tmp_path, needed=1 << 60)
    cfg = make_config(project, tmp_path / "ds", splits={"train": {"count": 10**12}})
    with pytest.raises(ValueError, match="not enough disk space"):
        generate_dataset(cfg, progress=False)
    assert not (tmp_path / "ds" / "images").exists()


def test_a_missing_train_config_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit, match="train config not found"):
        main(["train", "-t", str(tmp_path / "typo.yaml"), "--data", str(tmp_path / "d.yaml")])


def test_default_device_runs_anywhere():
    assert TrainConfig().device == "cpu"
