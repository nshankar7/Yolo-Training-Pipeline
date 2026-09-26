"""The one-command pipeline and stress-set evaluation, with Ultralytics faked out."""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest
import yaml
from conftest import make_config

from arrowgen import evaluate as eval_mod
from arrowgen import train as train_mod
from arrowgen.cli import _ingest_is_stale, main
from arrowgen.evaluate import format_table, prepare_stress_sets
from arrowgen.writer import generate_dataset


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    calls = {"train": [], "eval": []}

    def fake_run_train(cfg, data_yaml, **overrides):
        save_dir = tmp_path / "runs" / f"train{len(calls['train'])}"
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"weights")
        calls["train"].append(data_yaml)
        return SimpleNamespace(results_dict={}), save_dir

    def fake_evaluate(weights, datasets, train_cfg):
        calls["eval"].append((weights, dict(datasets)))
        return [{"set": name, "mAP50": 0.5, "mAP50/loop_cw": 0.4} for name in datasets]

    monkeypatch.setattr(train_mod, "run_train", fake_run_train)
    monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)
    return calls


def write_config(project, tmp_path, name="gen.yaml", **overrides):
    cfg = make_config(project, tmp_path / "ds", **overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


def write_stress_preset(tmp_path, name="stress_tiny.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({
        "output": {"root": str(tmp_path / "stress")},
        "splits": {"train": {"count": 0}, "val": {"count": 2, "stream": 9}},
    }))
    return path


def test_a_val_only_set_names_an_existing_train_folder(project, tmp_path):
    cfg = make_config(project, tmp_path / "ds", splits={"train": {"count": 0}})
    generate_dataset(cfg, progress=False)
    data = yaml.safe_load((tmp_path / "ds" / "data.yaml").read_text())
    assert data["train"] == data["val"] == "images/val"
    assert (tmp_path / "ds" / data["train"]).is_dir()


def test_stress_sets_are_generated_once_and_reused(project, tmp_path):
    base = write_config(project, tmp_path)
    preset = write_stress_preset(tmp_path)
    first = prepare_stress_sets([base], presets=[preset], progress=False)
    assert first["stress_tiny"].exists()
    meta = tmp_path / "stress" / "manifest" / "val.meta.json"
    stamp = meta.stat().st_mtime_ns

    prepare_stress_sets([base], presets=[preset], progress=False)
    assert meta.stat().st_mtime_ns == stamp, "an unchanged stress set should be reused"
    prepare_stress_sets([base], ["seed=5"], presets=[preset], progress=False)
    assert meta.stat().st_mtime_ns != stamp, "changed settings must regenerate it"


def test_ingest_is_stale_until_processed_is_newer_than_raw(project, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(project["raw"], raw)
    processed = tmp_path / "processed"
    classes = tmp_path / "classes.yaml"
    shutil.copy(project["classes_file"], classes)
    cfg = make_config(project, tmp_path / "ds", processed_dir=str(processed),
                      classes_file=str(classes))
    assert _ingest_is_stale(cfg, raw), "nothing processed yet"

    shutil.copytree(project["processed"], processed)
    past = 1_000_000_000
    for p in [*raw.rglob("*"), classes]:
        if p.is_file():
            os.utime(p, (past, past))
    assert not _ingest_is_stale(cfg, raw)

    touched = next(p for p in raw.rglob("*.png"))
    os.utime(touched, None)  # edited just now
    assert _ingest_is_stale(cfg, raw)


def test_run_goes_end_to_end_and_skips_finished_steps(  # noqa: PLR0913
    project, tmp_path, fakes, capsys, monkeypatch
):
    cfg_path = write_config(project, tmp_path)
    preset = write_stress_preset(tmp_path)
    args = ["run", "-c", str(cfg_path), "-t", str(tmp_path / "none.yaml"), "--skip-ingest",
            "--quiet"]

    with pytest.raises(SystemExit, match="train config not found"):
        main(args)  # a bad train config fails before any generation
    assert not (tmp_path / "ds").exists()

    args[4] = str(tmp_path / "train.yaml")
    (tmp_path / "train.yaml").write_text("device: cpu\n")
    monkeypatch.setattr(eval_mod, "STRESS_GLOB", preset.name)
    monkeypatch.chdir(tmp_path)
    assert main([*args, "--stress"]) == 0
    assert main(args) == 0

    out = capsys.readouterr().out
    assert out.count("up to date: ") == 1, "the second run should reuse the dataset"
    assert len(fakes["train"]) == 2
    first_weights, first_sets = fakes["eval"][0]
    assert set(first_sets) == {"val", "stress_tiny"}
    assert first_weights.name == "best.pt"
    assert (first_weights.parent.parent / "eval.csv").exists()


def test_eval_table_lists_per_class_columns():
    table = format_table([{"set": "val", "mAP50": 0.5, "mAP50/loop_cw": 0.25}])
    assert "mAP50/loop_cw" in table.splitlines()[0]
    assert "0.250" in table

