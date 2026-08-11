from pathlib import Path

import pytest

from src.paths import dataset_dir, dataset_file, repository_root


def test_default_paths_use_examples_layout(monkeypatch):
    for variable in (
        "LLM_ST_DATA_ROOT",
        "LLM_ST_STARMAP_ROOT",
        "LLM_ST_VISIUM_ROOT",
        "LLM_ST_MERFISH_ROOT",
        "LLM_ST_COPD_ROOT",
        "LLM_ST_STARMAP_TEMPLATE",
    ):
        monkeypatch.delenv(variable, raising=False)
    root = repository_root()
    assert dataset_dir("starmap", "BZ5") == root / "examples/starmap/BZ5"
    assert dataset_dir("visium_libd", "151507") == root / "examples/visium_libd/151507"
    assert dataset_file("merfish", "MERFISH_25") == root / "examples/merfish/MERFISH_25.h5ad"
    assert dataset_dir("copd", "230267_Slide2") == root / "examples/copd/230267_Slide2"


def test_starmap_path_uses_public_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ST_STARMAP_ROOT", str(tmp_path))
    assert dataset_dir("starmap", "BZ5") == tmp_path / "STARmap_mouse_cortex_BZ5"


def test_merfish_file_uses_sample_name(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ST_MERFISH_ROOT", str(tmp_path))
    assert dataset_file("merfish", "MERFISH_25") == tmp_path / "MERFISH_25.h5ad"


def test_copd_rejects_nonpublication_slide(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ST_COPD_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="230267_Slide2"):
        dataset_dir("copd", "230267_Slide1")
