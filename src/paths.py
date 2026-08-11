"""Portable path resolution for the public SPEAK workflows.

All paths can be overridden with environment variables.  Defaults describe the
layout expected in the publication data package and never point to an author's
home directory.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_ALIASES = {
    "libd": "visium_libd",
    "visium": "visium_libd",
    "visium_libd": "visium_libd",
    "starmap": "starmap",
    "merfish": "merfish",
    "copd": "copd",
}

_ROOT_ENV = {
    "starmap": "LLM_ST_STARMAP_ROOT",
    "visium_libd": "LLM_ST_VISIUM_ROOT",
    "merfish": "LLM_ST_MERFISH_ROOT",
    "copd": "LLM_ST_COPD_ROOT",
}


def repository_root() -> Path:
    """Return the checked-out repository root."""

    return REPO_ROOT


def data_root() -> Path:
    """Return the root of the bundled publication examples."""

    return Path(os.environ.get("LLM_ST_DATA_ROOT", REPO_ROOT / "examples")).expanduser()


def examples_root() -> Path:
    """Return the bundled, directly usable publication-data directory."""

    return data_root()


def results_root() -> Path:
    """Return bundled row-level annotations and benchmark tables."""

    return Path(
        os.environ.get("LLM_ST_RESULTS_ROOT", examples_root() / "results")
    ).expanduser()


def intermediate_root() -> Path:
    """Return bundled expensive or model-audit intermediates."""

    return Path(
        os.environ.get(
            "LLM_ST_INTERMEDIATE_ROOT", examples_root() / "intermediates"
        )
    ).expanduser()


def run_root() -> Path:
    """Return the root used for generated requests, responses, and results."""

    return Path(os.environ.get("LLM_ST_RUN_ROOT", REPO_ROOT)).expanduser()


def dataset_root(dataset: str) -> Path:
    """Return the configured root directory for a supported dataset."""

    key = _ALIASES.get(dataset.lower())
    if key is None:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    defaults = {
        "starmap": examples_root() / "starmap",
        "visium_libd": examples_root() / "visium_libd",
        "merfish": examples_root() / "merfish",
        "copd": examples_root() / "copd" / "230267_Slide2",
    }
    return Path(os.environ.get(_ROOT_ENV[key], defaults[key])).expanduser()


def dataset_dir(dataset: str, sample: str | None = None) -> Path:
    """Resolve an analysis directory without embedding machine-specific paths."""

    key = _ALIASES.get(dataset.lower())
    if key is None:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    root = dataset_root(key)
    if sample is None:
        return root
    if key == "starmap":
        default_template = (
            "STARmap_mouse_cortex_{sample}"
            if _ROOT_ENV[key] in os.environ
            else "{sample}"
        )
        template = os.environ.get("LLM_ST_STARMAP_TEMPLATE", default_template)
        return root / template.format(sample=sample)
    if key == "copd":
        if sample != "230267_Slide2":
            raise ValueError("The public COPD workflow supports only 230267_Slide2")
        return root
    return root / sample


def dataset_file(dataset: str, sample: str, filename: str | None = None) -> Path:
    """Resolve a primary input file for a sample."""

    key = _ALIASES.get(dataset.lower())
    if key == "merfish":
        return dataset_root(key) / (filename or f"{sample}.h5ad")
    directory = dataset_dir(dataset, sample)
    if filename is None:
        raise ValueError("filename is required for directory-based datasets")
    return directory / filename


def run_path(category: str, *parts: str) -> Path:
    """Resolve a generated artifact path below ``LLM_ST_RUN_ROOT``."""

    return run_root() / category / Path(*parts)
