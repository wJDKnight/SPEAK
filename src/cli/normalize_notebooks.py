"""Strip outputs and replace author-machine paths in public notebooks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.paths import repository_root


IMPORT_LINE = "from src.paths import dataset_dir, dataset_file, dataset_root, repository_root\n"

PLAIN_REPLACEMENTS = {
    "model_config/": "configs/",
    "configs/gene_lists/starmap_representative_genes.txt": "examples/starmap/representative_genes.txt",
    "./all_results": "examples/results",
    "all_results/": "examples/results/",
    "./deconv_result": "examples/visium_libd/deconv_result",
    "./localllm_results": "examples/intermediates/local_llm_results",
    "localllm_results/": "examples/intermediates/local_llm_results/",
    "MERFISH_data/new_gene_sets.txt": "examples/merfish/gene_sets.tsv",
    "MERFISH_data/merged_MERFISH.rda": "examples/intermediates/merfish/merged_MERFISH.rda",
    "MERFISH_data/integrated_MERFISH.rda": "examples/intermediates/merfish/integrated_MERFISH.rda",
    "sample_2/selected cell/selected_cells_id.csv": "selected_cells/selected_cells_id.csv",
    "/Users/hw568/project/LLM_ST/": "./",
    "~/project/LLM_ST/": "./",
    "python -u submit_end2end.py": "python -u -m src.submit_end2end",
    "python -u src/submit_end2end.py": "python -u -m src.submit_end2end",
    "python -u src/retrive_batch_results_parallel.py": "python -u -m src.retrive_batch_results_parallel",
    "~/storage/reference_genome/marker_genes/": "data/reference/marker_genes/",
}

REGEX_REPLACEMENTS = [
    (r'(?<!examples/results/)kmeans_results/', 'examples/results/kmeans_results/'),
    (r'(?<!examples/)source_data/', 'examples/source_data/'),
    (r'(?<!examples/results/)test_p_results/', 'examples/results/test_p_results/'),
    (
        r'(?P<indent>\s*)config\.folder_path = f"\./batch_json/\{config\.data_name\}_\{config\.model_type\}"\n?',
        r'\g<indent>config.refresh_paths()\n',
    ),
    (
        r'(?P<indent>\s*)config\.output_path = f"\./batch_results/\{config\.data_name\}_\{config\.model_type\}"\n?',
        '',
    ),
    (
        r'representetive_gene_list = \["Aqp4", "Gad1", "Gad2", "Sst", "Rbp4", "Cux2", "Synpr", "Nos1", "Cpne5", "Bdnf", "Adcyap1", "Syt6", "Pcp4", "Sla", "Ctgf", "Foxp2"\](?:  # seurat cluster)?',
        'representetive_gene_list = (repository_root() / "examples/starmap/representative_genes.txt").read_text().splitlines()',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/STARmap_mouse_cortex_\{config\.data_name\}/"',
        'str(dataset_dir("starmap", config.data_name))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/STARmap_mouse_cortex_\{data_name\}/"',
        'str(dataset_dir("starmap", data_name))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/spatialLIBD/\{config\.data_name\}/"',
        'str(dataset_dir("visium_libd", config.data_name))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/spatialLIBD/\{data_name\}/"',
        'str(dataset_dir("visium_libd", data_name))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/MERFISH/\{config\.data_name\}\.h5ad"',
        'str(dataset_file("merfish", config.data_name))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/MERFISH/"',
        'str(dataset_root("merfish"))',
    ),
    (
        r'f"/Users/hw568/storage/collections_spatial_datasets/Xenium_COPD"',
        'str(dataset_dir("copd", "230267_Slide2"))',
    ),
    (
        r'"/Users/hw568/storage/collections_spatial_datasets/spatialLIBD/"',
        'str(dataset_root("visium_libd"))',
    ),
    (
        r'"/Users/hw568/storage/collections_spatial_datasets/spatialDLPFC_new/?"',
        '"data/processed_inputs/spatialDLPFC_new"',
    ),
    (
        r'"/Users/hw568/Downloads/MERFISH_[^"]+\.h5ad"',
        'str(dataset_file("merfish", "MERFISH_25"))',
    ),
    (
        r'"/Users/hw568/Library/CloudStorage/OneDrive-YaleUniversity/microEv/BASS-Analysis-master/BASS_Results/',
        '"data/processed_inputs/baselines/BASS_Results/',
    ),
    (
        r'"/Users/hw568/Library/CloudStorage/OneDrive-YaleUniversity/microEv/IRIS/',
        '"data/processed_inputs/baselines/IRIS/',
    ),
    (
        r'model_path="/Users/hw568/project/llama/[^"\n]+"',
        'model_path=os.environ["LLM_ST_LOCAL_MODEL"]',
    ),
]


def rewrite_source(text: str) -> str:
    if "representetive_gene_list = [" in text and all(
        marker in text for marker in ("'Fn1'", "'Slco1a4'", "'Cyr61'")
    ):
        indent = text[: len(text) - len(text.lstrip())]
        return (
            f'{indent}representetive_gene_list = (repository_root() / '
            '"examples/merfish/representative_genes.txt").read_text().splitlines()\n'
        )
    for old, new in PLAIN_REPLACEMENTS.items():
        text = text.replace(old, new)
    for pattern, replacement in REGEX_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


def normalize(path: Path, check: bool = False) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = json.dumps(notebook, sort_keys=True, ensure_ascii=False)
    if path.name == "plot_kmeans_r_spot.ipynb":
        paper_cells = []
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown" and "# new DLPFC" in source:
                break
            paper_cells.append(cell)
        notebook["cells"] = paper_cells
    first_code_cell = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if first_code_cell is None:
            first_code_cell = cell
        cell["outputs"] = []
        cell["execution_count"] = None
        cell["source"] = [rewrite_source(line) for line in cell.get("source", [])]

    if first_code_cell is not None and not any(
        "from src.paths import" in line for line in first_code_cell.get("source", [])
    ):
        first_code_cell["source"] = [IMPORT_LINE, *first_code_cell.get("source", [])]

    updated = json.dumps(notebook, sort_keys=True, ensure_ascii=False)
    changed = original != updated
    if changed and not check:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    paths = args.paths or sorted((root / "workflows").glob("*/*.ipynb")) + sorted(
        (root / "analysis").glob("*/*.ipynb")
    )
    changed = [path for path in paths if normalize(path, check=args.check)]
    print(f"{'Would normalize' if args.check else 'Normalized'} {len(changed)} notebook(s)")
    if args.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
