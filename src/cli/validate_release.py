"""Validate the compact code release and, optionally, the full data package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.paths import repository_root


ABSOLUTE_PATH = re.compile(r"(?:/Users/|/home/|/vast/|~/)")
STALE_REFERENCE = re.compile(r"(?:example_data/|model_config/|\bllmst-(?:evaluate|smoke-test|validate-release))")
PUBLIC_DIRS = ("src", "workflows", "analysis", "configs", "docs", "environment", "tests")
TEXT_SUFFIXES = {".py", ".r", ".R", ".sh", ".yaml", ".yml", ".md", ".txt", ".toml"}
STARMAP_SAMPLES = ("BZ5", "BZ9", "BZ14")
VISIUM_SAMPLES = (
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
)
MERFISH_SAMPLES = tuple(f"MERFISH_{index}" for index in range(25, 30))
LEGACY_PATHS = ("example_data", "model_config")


def source_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def public_text_paths(root: Path) -> list[Path]:
    paths = [
        root / "README.md",
        root / "zeroshot_notebook.ipynb",
        root / "examples/README.md",
    ]
    paths.extend((root / "examples").rglob("README.md"))
    for directory in PUBLIC_DIRS:
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.suffix == ".ipynb")
        )
    return sorted(set(paths))


def require_files(paths: list[Path], root: Path, errors: list[str], label: str) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            try:
                display = path.relative_to(root)
            except ValueError:
                display = path
            errors.append(f"required {label} is absent or empty: {display}")


def compact_release_files(root: Path) -> list[Path]:
    required = [
        root / "README.md",
        root / "LICENSE",
        root / "pyproject.toml",
        root / "fig1_new_withHC.png",
        root / "zeroshot_notebook.ipynb",
        root / "environment/environment.yml",
        root / "environment/.env.example",
        root / "docs/DATA_AVAILABILITY.md",
        root / "docs/DATA_LAYOUT.md",
        root / "docs/REPRODUCIBILITY.md",
        root / "examples/README.md",
        root / "examples/starmap/README.md",
        root / "examples/starmap/representative_genes.txt",
        root / "examples/source_data/README.md",
        root / "examples/source_data/noise_experiment.csv",
        root / "examples/source_data/runtime_reference.csv",
        root / "examples/source_data/ground_truth_continuity.csv",
    ]
    for sample in STARMAP_SAMPLES:
        required.extend(
            root / "examples/starmap" / sample / filename
            for filename in ("data.csv", "celltype.csv", "pos.csv", "domain.csv")
        )
    return required


def full_data_files(data_root: Path) -> list[Path]:
    required = [
        data_root / "starmap/representative_genes.txt",
        data_root / "merfish/gene_sets.tsv",
        data_root / "merfish/representative_genes.txt",
        data_root / "copd/230267_Slide2/sample_230267_Slide2_k20.csv",
        data_root / "copd/230267_Slide2/230267_Slide2_zeroshot_gpt4o_refined_k20.csv",
        data_root / "copd/230267_Slide2/230267_Slide2_finetune_gpt4o_refined_k20.csv",
        data_root / "copd/230267_Slide2/selected_cells/selected_cells_id.csv",
        data_root / "results/zeroshot_all_metrics_all_replicates.csv",
        data_root / "results/finetunePro_all_metrics_all_replicates.csv",
        data_root / "source_data/noise_experiment.csv",
        data_root / "source_data/runtime_reference.csv",
        data_root / "source_data/ground_truth_continuity.csv",
        data_root / "intermediates/merfish/merged_MERFISH.rda",
        data_root / "intermediates/merfish/integrated_MERFISH.rda",
    ]
    for sample in STARMAP_SAMPLES:
        required.extend(
            data_root / "starmap" / sample / filename
            for filename in ("data.csv", "celltype.csv", "pos.csv", "domain.csv")
        )
    for sample in VISIUM_SAMPLES:
        required.extend(
            (
                data_root / "visium_libd" / sample / "filtered_feature_bc_matrix.h5",
                data_root / "visium_libd" / sample / "metadata.tsv",
                data_root / "visium_libd" / sample / "spatial/tissue_positions_list.csv",
                data_root
                / "visium_libd"
                / sample
                / f"celltype_proportions_{sample}.csv",
            )
        )
    required.extend(data_root / "merfish" / f"{sample}.h5ad" for sample in MERFISH_SAMPLES)
    return required


def validate_full_data(data_root: Path, root: Path, errors: list[str]) -> None:
    require_files(full_data_files(data_root), root, errors, "full-data file")
    required_directories = (
        "results/zeroshot_starmap",
        "results/finetunePro_starmap",
        "results/zeroshot_visium",
        "results/finetunePro_visium",
        "results/zeroshot_merfish",
        "results/finetunePro_merfish",
        "intermediates/model_requests",
        "intermediates/model_responses",
        "intermediates/finetuning",
        "intermediates/local_llm_results",
        "intermediates/visium_libd_deconvolution",
    )
    for relative in required_directories:
        directory = data_root / relative
        if not directory.is_dir() or not any(path.is_file() for path in directory.rglob("*")):
            errors.append(f"required full-data directory is empty: {directory}")


def validate_notebooks(root: Path, errors: list[str]) -> None:
    for base in (root / "workflows", root / "analysis"):
        for path in base.rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    errors.append(
                        f"notebook retains execution state: {path.relative_to(root)} cell {index}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also reject legacy paths, caches, and notebook execution state",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="also validate the complete downloaded data package at this path",
    )
    args = parser.parse_args()

    root = repository_root()
    errors: list[str] = []
    require_files(compact_release_files(root), root, errors, "compact-release file")

    excluded_sources = {"validate_release.py", "normalize_notebooks.py"}
    for path in public_text_paths(root):
        if path.name in excluded_sources:
            continue
        text = source_text(path)
        if ABSOLUTE_PATH.search(text):
            errors.append(f"machine-specific path in {path.relative_to(root)}")
        if STALE_REFERENCE.search(text):
            errors.append(f"legacy path or CLI name in {path.relative_to(root)}")
        if "tests" not in path.parts and re.search(r"(?:230267_|sample_230267_)Slide1", text):
            errors.append(f"COPD Slide1 reference in {path.relative_to(root)}")

    if args.strict:
        for relative in LEGACY_PATHS:
            if (root / relative).exists():
                errors.append(f"legacy release path remains present: {relative}")
        for path in root.rglob(".DS_Store"):
            if ".git" in path.parts:
                continue
            errors.append(f"system artifact remains present: {path.relative_to(root)}")
        validate_notebooks(root, errors)

    if args.data_root is not None:
        data_root = args.data_root
        if not data_root.is_absolute():
            data_root = root / data_root
        validate_full_data(data_root.resolve(), root, errors)

    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        raise SystemExit(1)

    print("Compact release validation passed")
    if args.data_root is not None:
        print(f"Full data validation passed: {data_root.resolve()}")


if __name__ == "__main__":
    main()
