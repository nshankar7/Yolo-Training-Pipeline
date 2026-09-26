"""Reproducibility: an image is a pure function of (config, seed, stream, round, index)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from conftest import make_config

from arrowgen.sample import GenContext, make_sample
from arrowgen.writer import generate_dataset


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def test_same_index_gives_the_same_image_twice(ctx):
    a = make_sample(ctx, stream=0, index=11)
    b = make_sample(ctx, stream=0, index=11)
    assert digest(a.image) == digest(b.image)
    assert [lab.to_line() for lab in a.labels] == [lab.to_line() for lab in b.labels]
    assert a.jpeg_quality == b.jpeg_quality


def test_a_fresh_context_reproduces_the_same_image(project, tmp_path):
    cfg = make_config(project, tmp_path / "ds")
    first = make_sample(GenContext.build(cfg), stream=0, index=5)
    second = make_sample(GenContext.build(cfg), stream=0, index=5)
    assert digest(first.image) == digest(second.image)


def test_different_indices_give_different_images(ctx):
    hashes = {digest(make_sample(ctx, stream=0, index=i).image) for i in range(12)}
    assert len(hashes) == 12


def test_train_and_val_streams_never_collide(ctx):
    train = {digest(make_sample(ctx, stream=0, index=i).image) for i in range(12)}
    val = {digest(make_sample(ctx, stream=1, index=i).image) for i in range(12)}
    assert train.isdisjoint(val)


def test_rounds_produce_unseen_images(ctx):
    base = {digest(make_sample(ctx, stream=0, index=i).image) for i in range(8)}
    later = {digest(make_sample(ctx, stream=0, index=i, round_=1).image) for i in range(8)}
    assert base.isdisjoint(later)


def test_changing_the_seed_changes_everything(project, tmp_path):
    a = make_sample(GenContext.build(make_config(project, tmp_path / "a", seed=1)), 0, 0)
    b = make_sample(GenContext.build(make_config(project, tmp_path / "b", seed=2)), 0, 0)
    assert digest(a.image) != digest(b.image)


@pytest.mark.slow
def test_worker_count_does_not_affect_output(project, tmp_path):
    """Parallel generation must be byte-identical to serial generation."""
    cfg_serial = make_config(project, tmp_path / "serial")
    cfg_parallel = make_config(project, tmp_path / "parallel")
    generate_dataset(cfg_serial, workers=0, progress=False)
    generate_dataset(cfg_parallel, workers=2, progress=False)

    serial_files = sorted((tmp_path / "serial" / "images").rglob("*.jpg"))
    parallel_files = sorted((tmp_path / "parallel" / "images").rglob("*.jpg"))
    assert [p.name for p in serial_files] == [p.name for p in parallel_files]
    assert serial_files
    for a, b in zip(serial_files, parallel_files, strict=True):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs between worker counts"
        label_a = a.parents[2].parent / "labels" / a.parent.parent.name / a.parent.name
        assert (label_a / a.name.replace(".jpg", ".txt")).exists()
