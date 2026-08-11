# usage
# python -u two_in_one_finetunePro_libd.py <data_name> <replicate_name> <replicate_name_unconserved> <fine_tuned_data_name> <skip_first_stage>
import sys

if len(sys.argv) < 6:
    print("Usage: python -u two_in_one_finetunePro_libd.py <data_name> <replicate_name> <replicate_name_unconserved> <fine_tuned_data_name> <skip_first_stage>")
    sys.exit(1)

data_name = sys.argv[1]
replicate_name = sys.argv[2]
replicate_name_unconserved = sys.argv[3]
fine_tuned_data_name = sys.argv[4]
skip_first_stage = sys.argv[5].lower() == 'true'

# IMPORTANT!!!! check the config_finetunePro_libd.yaml

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

# %%
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
# first stage
if skip_first_stage:
    print("skip first stage")
else:
    # submit_end2end.py
    generate_json_end2end(neighbor_normalized_df, config, prompt_func=prompt.finetune_user_celltype_geneorder, n_rows=1, batch_size=3000, df_extra=neighbor_normalized_df_genes)

    import subprocess
    print('submiting ', data_name)
    cmd = f"python -u -m src.submit_end2end configs/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro{config.replicate}.out 2>&1"
    subprocess.run(cmd, check=True, text=True, shell=True)


# %%

# 初始化空列表以保存custom_id和content
gpt_results_df = pd.DataFrame()
n_batch = 2
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
                content = content.replace('\n-', " ").replace('```', "").replace('json', "").replace('plaintext', '').replace('python', '').replace('Output:', 'Outputs:').replace('\n', " ").replace("‘", "'").replace("’", "'")
                content = content.replace("Layer ", "Layer")
                # extract outputs
                extract_dict = extract_output_microenvironments(content)
                # extract_dict = extract_last_braces(content)

                # 将提取到的信息添加到数据框中
                gpt_results_df = pd.concat([gpt_results_df, pd.DataFrame(extract_dict.values(), index=[custom_id])], ignore_index=False)
                    
            except json.JSONDecodeError:
                print(f"无法解析JSON字符串: {line}")


# %% [markdown]
# # check the results


# %%
gpt_results_df = gpt_results_df[[0]]
gpt_results_df.columns = ['finetunePro_gpt4o_mini']
gpt_results_df.index = gpt_results_df.index.astype(str)


# %%
# gpt_results_df = pd.read_csv(f"./gpt4omini_results/{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv", index_col=0)
# gpt_results_df.index = gpt_results_df.index.astype(str)

# Replace strings in the first column of gpt_results_df that contain keywords plus '**' or '.'
import re

# Get the list of keywords from domain_mapping
keywords = list(domain_mapping.values())

# Create a regex pattern to capture any keyword possibly surrounded by other text
pattern = r'.*(' + '|'.join(map(re.escape, keywords)) + r')[\*\.\s]*.*'

gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == "LayerWM", "finetunePro_gpt4o_mini"] = "WM"
gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == " WM", "finetunePro_gpt4o_mini"] = "WM"


# Replace the entire string with the captured keyword only if it could be not unknown
for nichtype in gpt_results_df.finetunePro_gpt4o_mini.value_counts()[gpt_results_df.finetunePro_gpt4o_mini.value_counts()<5].index:
    gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"] = gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"].str.replace(pattern, r'\1', regex=True)

# %%
# if the number of the cell type is less than 4, set it to unknown
for nichtype in gpt_results_df.finetunePro_gpt4o_mini.value_counts()[gpt_results_df.finetunePro_gpt4o_mini.value_counts()<5].index:
    gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"] = "unknown"

# %%


# %% [markdown]
# ## plot and save

# %%
adata.obs = adata.obs.join(gpt_results_df)
# replace the NA in adata.obs['finetune_gpt4o'] with "unknown"
adata.obs['finetunePro_gpt4o_mini'] = adata.obs['finetunePro_gpt4o_mini'].fillna("unknown")
# sc.pl.spatial(adata, color="finetunePro_gpt4o_mini",library_id=config.data_name, title =  f"finetunePro_gpt4o_mini")
# print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini']))
# print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini']))

# refine the niche
config.r_factor = 1
refined_niche = relabel_cells(adj_matrix.toarray(), gpt_results_df['finetunePro_gpt4o_mini'])
adata.obs['finetunePro_gpt4o_mini_refined'] = refined_niche
# sc.pl.spatial(adata, color="finetunePro_gpt4o_mini_refined", library_id=config.data_name, title =  f"finetunePro_gpt4o_mini_refined")
# print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_refined']))
print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_refined']))


# %%
save_folder = "./finetune_results/LIBD"
adata.obs.to_csv(f"{save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

print(f"save first stage result to {save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

# %% [markdown]
# # high confidence cell

# %%
label_key = 'finetunePro_gpt4o_mini'  # IMPORTANT!!!!! before refinement
high_confidence_mask = find_high_confidence_cells(
    adata,
    label_key=label_key,
    k=min(30, int((np.mean(n_neighbors))/2)),  # Consider top 20/ half of the mean nearest neighbors  # IMPORTANT!!!!!!  others are min(20) only merfish29 is min(30)
    distance_threshold=config.r
)

adata.obs['confident_cells'] = adata.obs['finetunePro_gpt4o_mini'].copy()
adata.obs['confident_cells'] = adata.obs['confident_cells'].astype(str)
adata.obs.loc[~high_confidence_mask, 'confident_cells'] = 'unconfident'
# sc.pl.spatial(adata, color="confident_cells", library_id=config.data_name, title =  f"confident_cells")

# Get unique values excluding 'unconfident'
confident_unique = set(adata.obs.loc[high_confidence_mask, 'confident_cells'].unique()) - {'unconfident'}
unique_layers = set(adata.obs[label_key].unique()) - {'unknown'}

# Check if any elements are missing
missing_elements = unique_layers - confident_unique
old_one_shot_df = pd.DataFrame()
if len(missing_elements) > 0:
    print(f"Missing elements in confident cells: {missing_elements}")
    print("use the old one_shot_df for that niche")
    old_one_shot_df = one_shot_df.loc[list(missing_elements)]
    
else:
    print("No missing elements in confident cells") 

conserved_normalized_df = neighbor_normalized_df.loc[high_confidence_mask]
conserved_normalized_df_genes = neighbor_normalized_df_genes.loc[high_confidence_mask]

un_conserved_normalized_df = neighbor_normalized_df.loc[~high_confidence_mask]
un_conserved_normalized_df_genes = neighbor_normalized_df_genes.loc[~high_confidence_mask]   

print(f"find {len(conserved_normalized_df)} conserved cells")
print(f"find {len(un_conserved_normalized_df)} un-conserved cells")
print(f"conserved_json/{config.data_name}_{config.model_type}/" )
# IMPORTANT!!!!!
# prototype is from conserved cells

# calculate prototype
conserved_neighbor_df = conserved_normalized_df.join(conserved_normalized_df_genes).copy()
one_shot_df = pd.concat([adata.obs[label_key].loc[conserved_neighbor_df.index], conserved_neighbor_df], axis=1).groupby(label_key, observed=False).mean()
one_shot_df = one_shot_df.dropna()

# remove unknown
if 'unknown' in one_shot_df.index:
    one_shot_df = one_shot_df.drop(index='unknown')
if len(old_one_shot_df) > 0:
    one_shot_df = pd.concat([old_one_shot_df, one_shot_df], axis=0)

config.system_prompt = prompt.CP_celltype_geneorder(one_shot_df, config)
if config.with_negatives:
    # get top 3 genes for each row in one_shot_df
    neg_top_3_genes = one_shot_df[config.gene_names].apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    config.neg_top_3_genes = neg_top_3_genes

print(config.folder_path)
print(config.gpt_model)
config.replicate = replicate_name_unconserved  # "_rep2R100_unconserved" is for one shot, _rep2R100_finetune is for finetune
print(config.replicate)

# IMPORTANT!!!!!
# json is from un-conserved cells
generate_json_end2end(un_conserved_normalized_df, config, prompt_func=prompt.finetune_user_celltype_geneorder, max_completion_tokens=128, batch_size=3000, df_extra=un_conserved_normalized_df_genes)



# %%


# %%
# submit_end2end.py

import subprocess
cmd = f"python -u -m src.submit_end2end configs/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro_73{config.replicate}_{config.prototype_p}.out 2>&1"
subprocess.run(cmd, check=True, text=True, shell=True)


# %%


# %% [markdown]
# # check second result
# 

# %%
# 初始化空列表以保存custom_id和content
gpt_results_df = pd.DataFrame()
if len(un_conserved_normalized_df) > 3000:
    n_batch = 2
else:
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
gpt_results_df.columns = ['finetunePro_gpt4o_mini']
gpt_results_df.index = gpt_results_df.index.astype(str)
gpt_results_df.index = gpt_results_df.index.str.replace("id_", "")


# %%
# fill nan with unknown
gpt_results_df.fillna("unknown", inplace=True)

gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == "LayerWM", "finetunePro_gpt4o_mini"] = "WM"
gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == " WM", "finetunePro_gpt4o_mini"] = "WM"

# %%
# if the number of the cell type is less than 4, set it to unknown
for nichtype in gpt_results_df.finetunePro_gpt4o_mini.value_counts()[gpt_results_df.finetunePro_gpt4o_mini.value_counts()<4].index:
    gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"] = "unknown"

# %%


# %%
# update the finetunePro_gpt4o_mini with second stage results
adata.obs['finetunePro_gpt4o_mini_twostage'] = adata.obs[label_key].copy()
adata.obs['finetunePro_gpt4o_mini_twostage'] = adata.obs['finetunePro_gpt4o_mini_twostage'].astype(str)
adata.obs.loc[gpt_results_df.index, 'finetunePro_gpt4o_mini_twostage'] = gpt_results_df.finetunePro_gpt4o_mini
# sc.pl.spatial(adata, color="finetunePro_gpt4o_mini_twostage", library_id=config.data_name, title =  f"finetunePro_gpt4o_mini_twostage")
# print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_twostage']))
# print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_twostage']))
# refine the niche
adj_matrix, _ = sparse_adjacency(pos_data.loc[neighbor_normalized_df.index], threshold=r)

refined_niche = relabel_cells(adj_matrix.toarray(), adata.obs['finetunePro_gpt4o_mini_twostage'])
adata.obs['finetunePro_gpt4o_mini_twostage_refined'] = refined_niche
# sc.pl.spatial(adata, color="finetunePro_gpt4o_mini_twostage_refined", library_id=config.data_name, title =  f"finetunePro_gpt4o_mini_twostage_refined")
# print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_twostage_refined']))
print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_twostage_refined']))


# %%
save_folder = './twostage_results/LIBD'
adata.obs.to_csv(f"{save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")
print(f"save second stage result to {save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")
