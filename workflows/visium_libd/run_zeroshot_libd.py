# this script is used to run the zeroshot_libd
# when using submit_end2end.py, don't use nohup 
# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scipy

import openai
import ast
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

from src.paths import dataset_dir
from src.utils import *
import src.prompt as prompt
import sys
data_name = sys.argv[1]
replicate_name = sys.argv[2]
# %% [markdown]
# # data

# %%
important_marker_genes = ["Aqp4", "Hpcal1", "Pvalb", "Frem3", "Pcp4", "Krt17","Mobp", 
                          "Lamp5", "Rorb", "Fezf2", "Syt6", "Fa2h",
                          "Plp1", "Foxj1", "Gfap", "Cpne5", "Kcnip2",
                          "Bgn", "Cux2", "Etv1",
                          "Zmat4", "Rab3c"]
important_marker_genes = [gene.upper() for gene in important_marker_genes]

# %%
config = load_config("configs/config_zeroshot_libd.yaml")
config.data_name = data_name
config.refresh_paths()
name_truth = config.name_truth
config.replicate = replicate_name
# %%
# --- Load data ---
print(f"processing data: {config.data_name}")
data_path = str(dataset_dir("visium_libd", config.data_name))
adata = sc.read_visium(data_path)
adata.var_names_make_unique()

cell_proportion_data = pd.read_csv(dataset_dir("visium_libd", config.data_name) / f"celltype_proportions_{config.data_name}.csv", index_col=0)

adata.obs = adata.obs.join(cell_proportion_data)

# Normalize data
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.scale(adata)



common_genes = list(set(adata.var_names) & set(important_marker_genes))
adata = adata[:, common_genes]

# read the metadata
meta_data = pd.read_csv(os.path.join(data_path, "metadata.tsv"), sep="\t")
# merge the metadata to adata
adata.obs = adata.obs.merge(meta_data, left_index=True, right_index=True, how="left")
# Remove rows with NaN values in 'layer_guess'
adata = adata[~adata.obs[name_truth].isna()].copy()

# Verify that NaNs have been removed
remaining_nan_count = adata.obs[name_truth].isna().sum()
print(f"Remaining NaN values in '{name_truth}' after removal: {remaining_nan_count}")

# clean the cell ID to save token
# Rename the obs_names of adata
adata.obs_names = [f'spot_{i}' for i in range(len(adata.obs_names))]

pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
cell_proportion_data = adata.obs[cell_proportion_data.columns].copy()

# --- Compute adjacency matrix ---
r = config.r
adj_matrix, distances = sparse_adjacency(pos_data, threshold=r, add_diagonal=True)

n_neighbors = adj_matrix.sum(axis=1).A1  # .A1 converts to 1D numpy array
# Convert n_neighbors to a column vector for element-wise division
n_neighbors_col = n_neighbors.reshape(-1, 1)
print(f"Mean of n_neighbors: {np.mean(n_neighbors)}")

# --- Calculate neighbor counts ---
neighbor_count = adj_matrix.dot(cell_proportion_data)

# Perform element-wise division between neighbor_count and n_neighbors_col
neighbor_matrix_normalized = neighbor_count / n_neighbors_col

neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized, 
                              index=adata.obs_names,
                              columns=cell_proportion_data.columns)



# %%


# --- Calculate neighbor genes ---
neighbor_genes = adj_matrix.dot(adata.X)

# Perform element-wise division between neighbor_count and n_neighbors_col
neighbor_matrix_normalized_genes = neighbor_genes / n_neighbors_col

neighbor_normalized_df_genes = pd.DataFrame(neighbor_matrix_normalized_genes, 
                              index=adata.obs_names, 
                              columns=adata.var_names)

# %% [markdown]
# # prompt

# %%
unique_layers = adata.obs[name_truth].unique()
domain_mapping = {i: layer for i, layer in enumerate(unique_layers)}

cell_names_mapping ={'Astro': 'Astrocyte',
 'EndoMural': 'Endothelial and mural cells',
 'Excit_L2_3': 'Excitatory neuron layer 2/3',
 'Excit_L3': 'Excitatory neuron layer 3',
 'Excit_L3_4_5': 'Excitatory neuron layer 3/4/5',
 'Excit_L4': 'Excitatory neuron layer 4',
 'Excit_L5': 'Excitatory neuron layer 5',
 'Excit_L5_6': 'Excitatory neuron layer 5/6',
 'Excit_L6': 'Excitatory neuron layer 6',
 'Inhib': 'Inhibitory neuron',
 'Micro': 'Microglia',
 'OPC': 'Oligodendrocyte precursor cell',
 'Oligo': 'Oligodendrocyte'}

config.domain_mapping = domain_mapping
config.cell_names_mapping = cell_names_mapping


config.cell_names = neighbor_normalized_df.columns
config.gene_names = common_genes



# %%
print(prompt.zeroshot_celltype_geneorder(neighbor_normalized_df, neighbor_normalized_df_genes, rows=[1], config=config))

# %% [markdown]
# # GPT

# %%
print(config.data_name)
print(config.gpt_model)
print(config.replicate)

# %%
generate_json_end2end(neighbor_normalized_df, config, prompt.zeroshot_geneorder, batch_size = 5000, n_rows = 1)

# %%
# submit_end2end.py

import subprocess
cmd = f"python -u -m src.submit_end2end configs/config_zeroshot_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_zeroshot{config.replicate}.out 2>&1"
subprocess.run(cmd, check=True, text=True, shell=True)



# %%
# 初始化空列表以保存custom_id和content
gpt_results_df = pd.DataFrame()
n_batch = 1
for i in range(1,n_batch+1):
    save_name = f"response_{config.data_name}_{i}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.txt"
    output_file_name = f"{config.output_path}/{save_name}"
    # 打开文件并逐行读取
    with open(output_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 解析每一行的json字符串
                json_data = json.loads(line.strip())
                
                # 提取custom_id和content信息
                custom_id = json_data['custom_id']
                content = json_data['response']['body']['choices'][0]['message']['content']
                content = content.replace('\n-', " ").replace('```', "").replace('json', "").replace('plaintext', '').replace('python', '').replace('Output:', 'Outputs:').replace('\n', " ")
                content = content.replace("Layer ", "Layer")
                # extract outputs
                extract_dict = extract_output_microenvironments(content)
                # extract_dict = extract_last_braces(content)

                # 将提取到的信息添加到数据框中
                gpt_results_df = pd.concat([gpt_results_df, pd.DataFrame(extract_dict.values(), index=[custom_id])], ignore_index=False)
                    
            except json.JSONDecodeError:
                print(f"无法解析JSON字符串: {line}")
gpt_results_df.columns = ['zeroshot_gpt4o_mini']
gpt_results_df.index = gpt_results_df.index.astype(str)


# %%
# Replace strings in the first column of gpt_results_df that contain keywords plus '**' or '.'
import re

# Get the list of keywords from domain_mapping
keywords = list(domain_mapping.values())

# Create a regex pattern to capture any keyword possibly surrounded by other text
pattern = r'.*(' + '|'.join(map(re.escape, keywords)) + r')[\*\.\s]*.*'

# Replace the entire string with the captured keyword only if it could be not unknown
for nichtype in gpt_results_df.zeroshot_gpt4o_mini.value_counts()[gpt_results_df.zeroshot_gpt4o_mini.value_counts()<3].index:
    gpt_results_df.loc[gpt_results_df.zeroshot_gpt4o_mini == nichtype, "zeroshot_gpt4o_mini"] = gpt_results_df.loc[gpt_results_df.zeroshot_gpt4o_mini == nichtype, "zeroshot_gpt4o_mini"].str.replace(pattern, r'\1', regex=True)



# %%
# if the number of the cell type is less than 4, set it to unknown
for nichtype in gpt_results_df.zeroshot_gpt4o_mini.value_counts()[gpt_results_df.zeroshot_gpt4o_mini.value_counts()<4].index:
    gpt_results_df.loc[gpt_results_df.zeroshot_gpt4o_mini == nichtype, "zeroshot_gpt4o_mini"] = "unknown"




# %% [markdown]
# # plot and save

# %%
# adata.obs.drop(columns=['zeroshot_gemini'], inplace=True)
# adata.obs.drop(columns=['zeroshot_gpt4o_mini'], inplace=True)

# %%
# adata.obs = adata.obs.join(gemini_results_df)
adata.obs = adata.obs.join(gpt_results_df)
# adata.obs['zeroshot_gemini'] = adata.obs['zeroshot_gemini'].fillna("unknown")
adata.obs['zeroshot_gpt4o_mini'] = adata.obs['zeroshot_gpt4o_mini'].fillna("unknown")
# sc.pl.spatial(adata, color=['zeroshot_gpt4o_mini',  name_truth], library_id=config.data_name, size=1.4)
print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['zeroshot_gpt4o_mini']))
print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['zeroshot_gpt4o_mini']))


# %% [markdown]
# # refine the niche

# %%
adj_matrix, _ = sparse_adjacency(pos_data, threshold=config.r)
adata.obs['zeroshot_gpt4o_mini_refined'] = relabel_cells(adj_matrix.toarray(), adata.obs['zeroshot_gpt4o_mini'])
print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['zeroshot_gpt4o_mini_refined']))

adata.obs.to_csv(f"./gpt4omini_results/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

# %%
