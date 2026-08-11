# this is for submitting testing jobs (first stage) of finetunePro_libd
replicate_name = "_rep1R600"  # NOTE： this need to change
fine_tuned_data_name = "151507"

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from src.paths import dataset_dir
from src.utils import *
import src.prompt as prompt

import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
config = load_config("configs/config_finetunePro_libd.yaml")
config.data_name = fine_tuned_data_name 
config.refresh_paths()
name_truth = config.name_truth

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
# ## sample data

# %%
# 设定随机种子
seed = 42  # 你可以根据需要修改这个值

# 定义分割比例 p (比如 0.7 表示 70% 数据用于训练，30% 数据用于测试)
p = config.prototype_p

# 分割数据集为训练集和测试集
train_neighbor_normalized_df, val_neighbor_normalized_df = train_test_split(neighbor_normalized_df, 
                                                            test_size=1-p, 
                                                            random_state=seed,
                                                            stratify=adata.obs[config.name_truth]
                                                           )


train_neighbor_normalized_df_genes, val_neighbor_normalized_df_genes = train_test_split(neighbor_normalized_df_genes, 
                                                            test_size=1-p, 
                                                            random_state=seed,
                                                            stratify=adata.obs[config.name_truth]
                                                           )


# %%
# check sample distribution
adata.obs[config.name_truth].loc[train_neighbor_normalized_df.index].value_counts()

# %%
# prototype
# calculate prototype
train_neighbor_df = train_neighbor_normalized_df.join(train_neighbor_normalized_df_genes).copy()
one_shot_df = pd.concat([adata.obs[config.name_truth].loc[train_neighbor_df.index], train_neighbor_df], axis=1).groupby(config.name_truth, observed=False).mean()
print(one_shot_df.index)


# %%
# validation in finetune is not necessary
# # finetune train and val data
# _, val_for_finetune = train_test_split(val_neighbor_normalized_df, 
#                                                             test_size=0.1, 
#                                                             random_state=seed
#                                                            )

# adata.obs.loc[val_for_finetune.index, config.name_truth].value_counts()

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


# generate Comparison-based Prompt
config.system_prompt = prompt.CP_celltype_geneorder(one_shot_df, config)


# %%
print(config.system_prompt)


# %%
print(prompt.finetune_user_celltype_geneorder(train_neighbor_normalized_df, train_neighbor_normalized_df_genes, 1, config))
print(prompt.finetune_assistant(train_neighbor_normalized_df, 1, adata.obs[config.name_truth]))


# %% [markdown]
# # test
for data_name in ["151508", "151509", "151510"]:    # NOTE:  this need to change

    # %% [markdown]
    # ## load test data 
    # Prototype in prompt is based on training data

    # %%
    # add model name to config
    # config.gpt_model = "ft:gpt-4o-mini-2024-07-18:whh:finetune-merfish270-3:ARMu34Br"

    # add model name to config and reload config
    config = load_config("configs/config_finetunePro_libd.yaml")
    config.data_name = data_name
    config.refresh_paths()
    name_truth = config.name_truth



    # %% [markdown]
    # ### this is codes for loading LIBD data

    # %%
    # this is codes for loading LIBD data
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





    # %%
    # --- Compute adjacency matrix ---
    r = config.r
    adj_matrix, distances = sparse_adjacency(pos_data, threshold=r, add_diagonal=True)

    # --- Calculate neighbor counts ---
    neighbor_count = adj_matrix.dot(cell_proportion_data)
    n_neighbors = adj_matrix.sum(axis=1).A1  # .A1 converts to 1D numpy array
    print(f"Mean of n_neighbors: {np.mean(n_neighbors)}")
    # Convert n_neighbors to a column vector for element-wise division
    n_neighbors_col = n_neighbors.reshape(-1, 1)
    # Perform element-wise division between neighbor_count and n_neighbors_col
    neighbor_matrix_normalized = neighbor_count / n_neighbors_col

    neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized, 
                                index=adata.obs_names,
                                columns=cell_proportion_data.columns)

    # --- Calculate neighbor genes ---
    neighbor_genes = adj_matrix.dot(adata.X)

    # Perform element-wise division between neighbor_count and n_neighbors_col
    neighbor_matrix_normalized_genes = neighbor_genes / n_neighbors_col

    neighbor_normalized_df_genes = pd.DataFrame(neighbor_matrix_normalized_genes, 
                                index=adata.obs_names, 
                                columns=adata.var_names)


    # %% [markdown]
    # ## test GPT

    # %%
    config.replicate = replicate_name


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


    # generate Comparison-based Prompt
    config.system_prompt = prompt.CP_celltype_geneorder(one_shot_df, config)


    # %%
    print(config.data_name)
    print(config.replicate)
    print(config.gpt_model)

    # %%
    generate_json_end2end(neighbor_normalized_df, config, prompt_func=prompt.finetune_user_celltype_geneorder, n_rows=1, batch_size=3000, df_extra=neighbor_normalized_df_genes)

    # %%
    # submit_end2end.py

    import subprocess
    print('submiting ', data_name)
    cmd = f"nohup python -u -m src.submit_end2end configs/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro{config.replicate}.out 2>&1 &"
    subprocess.run(cmd, check=True, text=True, shell=True)
