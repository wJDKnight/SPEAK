"""Offline STARmap BZ5 smoke test for loading, neighbourhoods, prompts, and metrics."""

from __future__ import annotations

from src.data_loader import load_spatial_data_csv
from src.evaluation import evaluate_dataframe
from src.paths import dataset_dir, repository_root
from src.prompt import zeroshot_celltype
from src.utils import load_config, prepare_neighbor_data


def main() -> None:
    root = repository_root()
    config = load_config(str(root / "configs/config_zeroshot_starmap.yaml"))
    adata = load_spatial_data_csv(str(dataset_dir("starmap", "BZ5")), config=config)
    config.domain_mapping = {
        1: "Layer 1",
        2: "Layer 2/3",
        3: "Layer 5",
        4: "Layer 6",
    }
    config.cell_names_mapping = {
        name: name for name in adata.obs[config.celltype_name].dropna().unique()
    }
    neighbours, _, adjacency = prepare_neighbor_data(adata, config)
    prompt = zeroshot_celltype(neighbours, [0], config)
    if not prompt or adjacency.shape[0] != adata.n_obs:
        raise RuntimeError("STARmap smoke test failed to create neighbourhood prompt data")

    metric_frame = adata.obs.rename(columns={config.name_truth: "truth"}).copy()
    metric_frame["prediction"] = metric_frame["truth"]
    metrics = evaluate_dataframe(metric_frame, "truth", "prediction")
    if metrics["ARI"] != 1.0 or metrics["NMI"] != 1.0:
        raise RuntimeError("STARmap identity-label metric smoke test failed")
    print(
        f"STARmap BZ5 smoke test passed: {adata.n_obs} cells, "
        f"{adjacency.nnz} adjacency entries, {len(prompt)} prompt characters"
    )


if __name__ == "__main__":
    main()
