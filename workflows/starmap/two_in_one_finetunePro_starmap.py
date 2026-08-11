# usage
# python -u two_in_one_finetunePro_libd.py <data_name> <replicate_name> <replicate_name_unconserved> <fine_tuned_data_name> <skip_first_stage>
import sys

if len(sys.argv) < 6:
    print("Usage: python -u two_in_one_finetune_starmap.py <data_name> <replicate_name> <replicate_name_unconserved> <fine_tuned_data_name> <skip_first_stage>")
    sys.exit(1)

data_name = sys.argv[1]
replicate_name = sys.argv[2]
replicate_name_unconserved = sys.argv[3]
fine_tuned_data_name = sys.argv[4]
skip_first_stage = sys.argv[5].lower() == 'true'

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
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
config = load_config("configs/config_finetunePro_starmap.yaml")
config.data_name = fine_tuned_data_name
config.refresh_paths()

# %%
# --- Load data ---
print(f"Loading data for {config.data_name}")
data_path = str(dataset_dir("starmap", config.data_name))
x_data_name = "data.csv"  
index_col = 0
adata = sc.read_csv(f"{data_path}/{x_data_name}", first_column_names=True)
celltype_data = pd.read_csv(f"{data_path}/celltype.csv", index_col=index_col)  # Assuming first column is index
celltype_data.columns = ["cell_type"] 
pos_data = pd.read_csv(f"{data_path}/pos.csv", index_col=index_col)
pos_data.columns = ['x', 'y']
domain_data = pd.read_csv(f"{data_path}/domain.csv", index_col=index_col)
domain_data.columns = [config.name_truth]
adata.obs = adata.obs.join([celltype_data, pos_data, domain_data])

# clean the cell ID to save token
adata.obs_names = list(range(len(adata)))
adata.obs_names = adata.obs_names.astype(str)

# rename the domain
domain_mapping = {1 : "Layer 1", 2 : "Layer 2/3", 3 : "Layer 5", 4 : "Layer 6"}
adata.obs[config.name_truth] = adata.obs[config.name_truth].map(domain_mapping)

pos_data = adata.obs[['x', 'y']]
celltype_data = adata.obs[['cell_type']]
domain_data = adata.obs[[config.name_truth]]

# --- Compute adjacency matrix ---
r = config.r
adj_matrix, distances = sparse_adjacency(pos_data, threshold=r)
# add diagonal to the adj_matrix
adj_matrix = adj_matrix + scipy.sparse.diags(np.ones(adj_matrix.shape[0]))

# --- Generate one-hot encoded matrix ---
one_hot_df = pd.get_dummies(celltype_data['cell_type'], prefix='').astype(int)
one_hot_matrix = one_hot_df.values
one_hot_matrix = csr_matrix(one_hot_matrix)  # Convert to sparse matrix

# --- Calculate neighbor counts ---
neighbor_count = adj_matrix.dot(one_hot_matrix)
n_neighbors = adj_matrix.sum(axis=1).A1  # .A1 converts to 1D numpy array
print(f"Mean number of neighbors: {np.mean(n_neighbors)}")
# Convert n_neighbors to a column vector for element-wise division
n_neighbors_col = n_neighbors.reshape(-1, 1)
# Perform element-wise division between neighbor_count and n_neighbors_col
neighbor_matrix_normalized = neighbor_count / n_neighbors_col

neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized.toarray(), 
                              index=celltype_data.index, 
                              columns=one_hot_df.columns.str.lstrip('_'))



# %%


# %% [markdown]
# ## sample data

# %%
# 设定随机种子
seed = 42  # 你可以根据需要修改这个值

# 定义分割比例 p (比如 0.7 表示 70% 数据用于训练，30% 数据用于测试)
p = 0.3

# 分割数据集为训练集和测试集
train_neighbor_normalized_df, val_neighbor_normalized_df = train_test_split(neighbor_normalized_df, 
                                                            test_size=1-p, 
                                                            random_state=seed,
                                                            stratify=domain_data[config.name_truth]
                                                           )






# %%
train_neighbor_df = train_neighbor_normalized_df
one_shot_df = pd.concat([adata.obs[config.name_truth].loc[train_neighbor_df.index], train_neighbor_df], axis=1).groupby(config.name_truth, observed=False).mean()
print(one_shot_df.index)




# %% [markdown]
# # prompt

# %%
domain_mapping = {1 : "Layer 1", 2 : "Layer 2/3", 3 : "Layer 5", 4 : "Layer 6"}

cell_names_mapping = {'Astro': 'Astrocytes',
 'Endo': 'Endothelial cells',
 'L5-1': 'Layer 5 pyramidal neuron subtype 1',
 'Lhx6': 'Lhx6-expressing interneurons',
 'NPY': 'Neuropeptide Y-expressing interneurons',
 'Oligo': 'Oligodendrocytes',
 'Reln': 'Reelin-expressing cells',
 'SST': 'Somatostatin-expressing interneurons',
 'Smc': 'Smooth muscle cells',
 'VIP': 'Vasoactive intestinal peptide-expressing interneurons',
 'eL2/3': 'Excitatory neuron layer 2/3',
 'eL5-2': 'Excitatory neuron layer 5 subtype 2',
 'eL5-3': 'Excitatory neuron layer 5 subtype 3',
 'eL6-1': 'Excitatory neuron layer 6 subtype 1',
 'eL6-2': 'Excitatory neuron layer 6 subtype 2'}

config.domain_mapping = domain_mapping
config.cell_names_mapping = cell_names_mapping
config.system_prompt = prompt.CP_celltype(one_shot_df,config)

 

# # test

# %% [markdown]
# ## load test data BZ9 BZ14 
# Prototype in prompt is based on training data

# %%
config = load_config("configs/config_finetunePro_starmap.yaml")
config.data_name = data_name
config.replicate = replicate_name
config.refresh_paths()


domain_mapping = {1 : "Layer 1", 2 : "Layer 2/3", 3 : "Layer 5", 4 : "Layer 6"}
cell_names_mapping = {'Astro': 'Astrocytes',
 'Endo': 'Endothelial cells',
 'L5-1': 'Layer 5 pyramidal neuron subtype 1',
 'Lhx6': 'Lhx6-expressing interneurons',
 'NPY': 'Neuropeptide Y-expressing interneurons',
 'Oligo': 'Oligodendrocytes',
 'Reln': 'Reelin-expressing cells',
 'SST': 'Somatostatin-expressing interneurons',
 'Smc': 'Smooth muscle cells',
 'VIP': 'Vasoactive intestinal peptide-expressing interneurons',
 'eL2/3': 'Excitatory neuron layer 2/3',
 'eL5-2': 'Excitatory neuron layer 5 subtype 2',
 'eL5-3': 'Excitatory neuron layer 5 subtype 3',
 'eL6-1': 'Excitatory neuron layer 6 subtype 1',
 'eL6-2': 'Excitatory neuron layer 6 subtype 2'}

config.domain_mapping = domain_mapping
config.cell_names_mapping = cell_names_mapping
config.system_prompt = prompt.CP_celltype(one_shot_df,config)


# %%
# --- Load data ---
data_path = str(dataset_dir("starmap", config.data_name))
x_data_name = "data.csv"  
index_col = 0
adata = sc.read_csv(f"{data_path}/{x_data_name}", first_column_names=True)
celltype_data = pd.read_csv(f"{data_path}/celltype.csv", index_col=index_col)  # Assuming first column is index
celltype_data.columns = ["cell_type"] 
pos_data = pd.read_csv(f"{data_path}/pos.csv", index_col=index_col)
pos_data.columns = ['x', 'y']
domain_data = pd.read_csv(f"{data_path}/domain.csv", index_col=index_col)
domain_data.columns = [config.name_truth]
adata.obs = adata.obs.join([celltype_data, pos_data, domain_data])

# clean the cell ID to save token
adata.obs_names = list(range(len(adata)))
adata.obs_names = adata.obs_names.astype(str)

# rename the domain
domain_mapping = {1 : "Layer 1", 2 : "Layer 2/3", 3 : "Layer 5", 4 : "Layer 6"}
adata.obs[config.name_truth] = adata.obs[config.name_truth].map(domain_mapping)

pos_data = adata.obs[['x', 'y']]
celltype_data = adata.obs[['cell_type']]
domain_data = adata.obs[[config.name_truth]]

# --- Compute adjacency matrix ---
r = config.r
adj_matrix, distances = sparse_adjacency(pos_data, threshold=r)
# add diagonal to the adj_matrix
adj_matrix = adj_matrix + scipy.sparse.diags(np.ones(adj_matrix.shape[0]))

# --- Generate one-hot encoded matrix ---
one_hot_df = pd.get_dummies(celltype_data['cell_type'], prefix='').astype(int)
one_hot_matrix = one_hot_df.values
one_hot_matrix = csr_matrix(one_hot_matrix)  # Convert to sparse matrix

# --- Calculate neighbor counts ---
neighbor_count = adj_matrix.dot(one_hot_matrix)
n_neighbors = adj_matrix.sum(axis=1).A1  # .A1 converts to 1D numpy array
# Convert n_neighbors to a column vector for element-wise division
n_neighbors_col = n_neighbors.reshape(-1, 1)
# Perform element-wise division between neighbor_count and n_neighbors_col
neighbor_matrix_normalized = neighbor_count / n_neighbors_col

neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized.toarray(), 
                              index=celltype_data.index, 
                              columns=one_hot_df.columns.str.lstrip('_'))


# %% [markdown]
# ## test GPT

# %%
generate_json_end2end(neighbor_normalized_df, config, prompt_func=prompt.finetune_user_deconv, n_rows=1, batch_size=5000)

# %%


# first stage
if skip_first_stage:
    print("skip first stage")
else:
    import subprocess
    print('submiting ', config.data_name)
    cmd = f"python -u -m src.submit_end2end configs/config_finetunePro_starmap.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro_BZ5.out 2>&1"
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
                content = content.replace('\n-', " ").replace('```', "").replace('json', "").replace('plaintext', '').replace('python', '').replace('Output:', 'Outputs:').replace('\n', " ").replace("‘", "'").replace("’", "'")
                content = content.replace("Layer ", "Layer")
                # extract outputs
                extract_dict = extract_output_microenvironments(content)
                # extract_dict = extract_last_braces(content)

                # 将提取到的信息添加到数据框中
                gpt_results_df = pd.concat([gpt_results_df, pd.DataFrame(extract_dict.values(), index=[custom_id])], ignore_index=False)
                    
            except json.JSONDecodeError:
                print(f"无法解析JSON字符串: {line}")

gpt_results_df = gpt_results_df[[0]]
gpt_results_df.columns = ['finetunePro_gpt4o_mini']
gpt_results_df.index = gpt_results_df.index.astype(str)

# %%






# fill nan with unknown
gpt_results_df.fillna("unknown", inplace=True)
# if the number of the cell type is less than 4, set it to unknown
for nichtype in gpt_results_df.finetunePro_gpt4o_mini.value_counts()[gpt_results_df.finetunePro_gpt4o_mini.value_counts()<4].index:
    gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"] = "unknown"

# %%
adata.obs = adata.obs.join(gpt_results_df)
# replace the NA in adata.obs['finetunePro_gpt4o'] with "unknown"
adata.obs['finetunePro_gpt4o_mini'] = adata.obs['finetunePro_gpt4o_mini'].fillna("unknown")




# %% [markdown]
# ## refine

# %%
# refine the niche
config.r_factor = 1
refined_niche = relabel_cells(adj_matrix.toarray(), gpt_results_df['finetunePro_gpt4o_mini'])
adata.obs['finetunePro_gpt4o_mini_refined'] = refined_niche
# sc.pl.spatial(adata, color="finetunePro_gpt4o_mini_refined", library_id=config.data_name, title =  f"finetunePro_gpt4o_mini_refined")
# print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_refined']))
print(normalized_mutual_info_score(adata.obs[config.name_truth], adata.obs['finetunePro_gpt4o_mini_refined']))


# %%
save_folder = "./finetune_results/STARmap"
adata.obs.to_csv(f"{save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

print(f"save first stage result to {save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")


# %% [markdown]
# # load first stage result

# %%


# %% [markdown]
# # high confidence cell

# %%
label_key = 'finetunePro_gpt4o_mini'  # IMPORTANT!!!!! before refinement
high_confidence_mask = find_high_confidence_cells(
    adata,
    pos_data = pos_data,
    label_key=label_key,
    k=min(30, int((np.mean(n_neighbors))/2)),  # Consider top 20/ half of the mean nearest neighbors  # IMPORTANT!!!!!!  others are min(20) only merfish29 is min(30)
    distance_threshold=config.r
)

adata.obs['confident_cells'] = adata.obs['finetunePro_gpt4o_mini'].copy()
adata.obs['confident_cells'] = adata.obs['confident_cells'].astype(str)
adata.obs.loc[~high_confidence_mask, 'confident_cells'] = 'unconfident'


# %%
# sc.pl.scatter(adata, x="x", y="y", color="confident_cells", title =  f"confident_cells")

# %%
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
# conserved_normalized_df_genes = neighbor_normalized_df_genes.loc[high_confidence_mask]

un_conserved_normalized_df = neighbor_normalized_df.loc[~high_confidence_mask]
# un_conserved_normalized_df_genes = neighbor_normalized_df_genes.loc[~high_confidence_mask]   

print(f"find {len(conserved_normalized_df)} conserved cells")
print(f"find {len(un_conserved_normalized_df)} un-conserved cells")

# calculate prototype
conserved_neighbor_df = conserved_normalized_df.copy()
one_shot_df = pd.concat([adata.obs[label_key].loc[conserved_neighbor_df.index], conserved_neighbor_df], axis=1).groupby(label_key, observed=False).mean()
one_shot_df = one_shot_df.dropna()

# remove unknown
if 'unknown' in one_shot_df.index:
    one_shot_df = one_shot_df.drop(index='unknown')
if len(old_one_shot_df) > 0:
    one_shot_df = pd.concat([old_one_shot_df, one_shot_df], axis=0)

# update config.domain_mapping
print("old one_shot_df.index", config.domain_mapping)
print("new one_shot_df.index", one_shot_df.index)
print("update config.domain_mapping")
config.domain_mapping = {i+1: one_shot_df.index[i] for i in range(len(one_shot_df))}
print("new config.domain_mapping", config.domain_mapping)

config.system_prompt = prompt.CP_celltype(one_shot_df,config)
if config.with_negatives:
    # get top 3 genes for each row in one_shot_df
    neg_top_3_genes = one_shot_df[config.gene_names].apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    config.neg_top_3_genes = neg_top_3_genes

print(config.folder_path)
print(config.gpt_model)
config.replicate = replicate_name_unconserved  # "_rep2R100_unconserved" is for one shot, _rep2R100_finetune is for finetune
print(config.replicate)


# %%
# json is from un-conserved cells
generate_json_end2end(un_conserved_normalized_df, config, prompt_func=prompt.finetune_user_deconv, max_completion_tokens=128, batch_size=3000)


# %%
# submit_end2end.py

import subprocess
cmd = f"python -u -m src.submit_end2end configs/config_finetunePro_starmap.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_{config.model_type}_{config.replicate}.out 2>&1"

subprocess.run(cmd, check=True, text=True, shell=True)


# %%
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
# if the number of the cell type is less than 4, set it to unknown
for nichtype in gpt_results_df.finetunePro_gpt4o_mini.value_counts()[gpt_results_df.finetunePro_gpt4o_mini.value_counts()<4].index:
    gpt_results_df.loc[gpt_results_df.finetunePro_gpt4o_mini == nichtype, "finetunePro_gpt4o_mini"] = "unknown"


# %%
# update the finetune_gpt4o_mini with second stage results
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
save_folder = './twostage_results/STARmap'
adata.obs.to_csv(f"{save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")
print(f"save second stage result to {save_folder}/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")


# %%

