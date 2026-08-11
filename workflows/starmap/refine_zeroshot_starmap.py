# %% [markdown]
# # read data

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy

from src.paths import dataset_dir, results_root
from src.utils import *
import src.prompt as prompt

data_name_list = ["BZ5", "BZ9", "BZ14"]
gpt_model_list = ["gpt4o", "gpt4o_mini", "gemini"]
reps = ["_rep1", "_rep2", "_rep3", "_rep4", "_rep5"]

for data_name in data_name_list:
    for gpt_model in gpt_model_list:
        for rep in reps:
            print(f"Processing {data_name} with {gpt_model}{rep}")
            
            label_key = f"zeroshot_{gpt_model}"

            # Load configuration
            config = load_config("configs/config_BZ9_zeroshot.yaml")
            config.data_name = data_name
            config.refresh_paths()
            name_truth = config.name_truth
            config.replicate = rep

            # Define paths
            result_dir = results_root() / "zeroshot_starmap"
            results_path = result_dir / "raw" / f"{gpt_model}_results" / f"{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv"
            save_dir = result_dir / "refined" / f"{gpt_model}_results"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv"

            # Skip if results file doesn't exist
            if not os.path.exists(results_path):
                print(f"Skipping {results_path} - file not found")
                continue

            try:
                # Load data
                data_path = str(dataset_dir("starmap", config.data_name))
                x_data_name = "data.csv"  
                index_col = 0
                adata = sc.read_csv(f"{data_path}/{x_data_name}", first_column_names=True)
                celltype_data = pd.read_csv(f"{data_path}/celltype.csv", index_col=index_col)
                celltype_data.columns = ["cell_type"] 
                pos_data = pd.read_csv(f"{data_path}/pos.csv", index_col=index_col)
                pos_data.columns = ['x', 'y']
                domain_data = pd.read_csv(f"{data_path}/domain.csv", index_col=index_col)
                domain_data.columns = ["niche_truth"]
                adata.obs = adata.obs.join([celltype_data, pos_data, domain_data])

                # Clean cell ID
                adata.obs_names = list(range(len(adata)))
                adata.obs_names = adata.obs_names.astype(str)

                pos_data = adata.obs[['x', 'y']]
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
