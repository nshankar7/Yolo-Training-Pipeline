"""Writing a dataset Ultralytics can actually read."""

from __future__ import annotations

import json

import cv2
import pytest
import yaml
from conftest import make_config

from arrowgen.preview import preview_dataset, preview_generated
from arrowgen.sample import GenContext
from arrowgen.stats import report
from arrowgen.writer import generate_dataset, image_path, label_path


@pytest.fixture
def dataset(project, tmp_path) -> dict:
    cfg = make_config(project, tmp_path / "ds")
    summary = generate_dataset(cfg, workers=0, progress=False)
    return {"cfg": cfg, "root": tmp_path / "ds", "summary": summary}


def test_label_path_mirrors_the_image_path():
    img = image_path(__import__("pathlib").Path("/data/set"), "train", 12345, 10000)
    assert img.as_posix().endswith("images/train/0001/train_00012345.jpg")
    assert label_path(img).as_posix().endswith("labels/train/0001/train_00012345.txt")


def test_every_image_has_a_label_file(dataset):
    root = dataset["root"]
    images = sorted(root.rglob("images/**/*.jpg"))
    assert len(images) == 6  # 4 train + 2 val
    for img in images:
        assert label_path(img).exists(), f"missing label file for {img.name}"
        assert cv2.imread(str(img)) is not None


def test_data_yaml_is_absolute_and_ordered(dataset):
    """A relative path here would be resolved against Ultralytics' own datasets folder."""
    data = yaml.safe_load((dataset["root"] / "data.yaml").read_text())
    assert data["path"].startswith("/")
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    names = GenContext.build(dataset["cfg"]).classes.names
    assert data["names"] == dict(enumerate(names))


def test_manifest_and_metadata_describe_the_run(dataset):
    root = dataset["root"]
    lines = (root / "manifest" / "train.jsonl").read_text().splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert [r["index"] for r in records] == [0, 1, 2, 3], "manifest should stay ordered"
    for record in records:
        assert record["n_labels"] == len(record["objects"])
        assert (root / record["image"]).exists()

    meta = json.loads((root / "dataset_meta.json").read_text())
    assert meta["config_hash"]
    assert meta["classes"] == GenContext.build(dataset["cfg"]).classes.names
    assert meta["summary"]["splits"]["train"]["images"] == 4


def test_regenerating_over_existing_files_needs_overwrite(dataset):
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_dataset(dataset["cfg"], workers=0, progress=False)
    generate_dataset(dataset["cfg"], workers=0, progress=False, overwrite=True)


def test_stats_report_mentions_every_class(dataset):
    text = report(dataset["root"], "train")
    for name in GenContext.build(dataset["cfg"]).classes.names:
        assert name in text
    assert "rejected placements" in text


def test_previews_render_from_both_sources(dataset, tmp_path):
    ctx = GenContext.build(dataset["cfg"])
    generated = preview_generated(ctx, 4, stream=0, start=0, role="train",
                                  out_path=tmp_path / "gen.jpg", cols=2)
    from_disk = preview_dataset(dataset["root"], "train", 4, ctx.classes.names,
                                tmp_path / "disk.jpg", cols=2)
    assert cv2.imread(str(generated)) is not None
    assert cv2.imread(str(from_disk)) is not None
