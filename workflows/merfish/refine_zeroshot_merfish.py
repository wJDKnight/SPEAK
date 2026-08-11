# %% [markdown]
# # read data

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy

from src.paths import dataset_file, results_root
from src.utils import *
import src.prompt as prompt

data_name_list = ["MERFISH_25", "MERFISH_26", "MERFISH_27", "MERFISH_28", "MERFISH_29"]
gpt_model_list = ["gpt4o_mini",  "gemini"]
reps = ["_rep1"]

for data_name in data_name_list:
    for gpt_model in gpt_model_list:
        for rep in reps:
            print(f"Processing {data_name} with {gpt_model}{rep}")
            
            label_key = f"zeroshot_{gpt_model}"

            # Load configuration
            config = load_config("configs/config_zeroshot_merfish.yaml")
            config.data_name = data_name
            config.refresh_paths()
            name_truth = config.name_truth
            config.replicate = rep

            # Define paths
            result_dir = results_root() / "zeroshot_merfish"
            results_path = result_dir / "raw" / f"{gpt_model}_results" / f"{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv"
            save_dir = result_dir / "refined" / f"{gpt_model}_results"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv"

            # Skip if results file doesn't exist
            if not os.path.exists(results_path):
                print(f"Skipping {results_path} - file not found")
                continue

            try:
                # --- Load data ---
                data_path = str(dataset_file("merfish", config.data_name))
                adata = sc.read_h5ad(data_path) 
                # rename the column of cell_class to cell_type
                adata.obs.rename(columns={'cell_class': 'cell_type'}, inplace=True)

                # clean the cell ID to save token
                adata.obs_names = list(range(len(adata)))
                adata.obs_names = adata.obs_names.astype(str)

                # Remove rows with NaN values in 'layer_guess'
                adata = adata[~adata.obs[name_truth].isna()].copy()

                # Verify that NaNs have been removed
                remaining_nan_count = adata.obs[name_truth].isna().sum()
                print(f"Remaining NaN values in '{name_truth}' after removal: {remaining_nan_count}")

                pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
                adata.obs = adata.obs.join(pos_data)
                # rename cell types
                celltype_rename = {
                    'Astrocyte' : 'Astrocyte',
                    'Endothelial 1': 'Endothelial',
                    'OD Mature 2': 'Mature oligodendrocytes',
                    'Inhibitory': 'Inhibitory',
                    'OD Immature 1': 'Immature oligodendrocytes',
                    'Excitatory': 'Excitatory',
                    'Endothelial 3': 'Endothelial',
                    'Microglia': 'Microglia',
                    'OD Mature 1': 'Mature oligodendrocytes',
                    'Pericytes': 'Pericytes',
                    'OD Mature 4': 'Mature oligodendrocytes',
                    'Endothelial 2': 'Endothelial',
                    'OD Mature 3': 'Mature oligodendrocytes',
                    'OD Immature 2': 'Immature oligodendrocytes',
                    'Ependymal': 'Ependymal'
                }
                adata.obs['cell_type'] = adata.obs['cell_type'].map(celltype_rename)
                celltype_data = adata.obs[['cell_type']]

                # Compute adjacency matrix
                r = config.r
                adj_matrix, distances = sparse_adjacency(pos_data, threshold=r)


                # Load GPT results and refine
                gpt_results_df = pd.read_csv(results_path, index_col=0)
                gpt_results_df.index = gpt_results_df.index.astype(str)

                adata.obs = adata.obs.join(gpt_results_df)

                # Refine and save
                refined_niche = relabel_cells(adj_matrix.toarray(), adata.obs[label_key])
                adata.obs[f"{label_key}_refined"] = refined_niche
                adata.obs.to_csv(save_path)
                print(f"Saved refined results to {save_path}")

            except Exception as e:
                print(f"Error processing {data_name} with {gpt_model}{rep}: {str(e)}")
                continue

print("Refinement process completed for all combinations")
