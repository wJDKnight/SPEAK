from math import floor, log10
import time
import re
import ast
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass, field
import os
import yaml
import pickle
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from scipy.spatial.distance import cdist
import scipy
from sklearn.metrics import adjusted_rand_score
from typing import Optional, List, Dict
from pydantic import BaseModel
from enum import Enum
from openai.lib._pydantic import to_strict_json_schema
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
import seaborn as sns
from matplotlib.patches import Patch

@dataclass
class Config:
    data_name: str
    r: float
    tissue_region: str
    model_type: str
    minimal_f: float   # minimal frequency of cell type in the neighborhood
    top_n: int
    Graph_type: str
    with_negatives: bool
    with_numbers: bool
    with_CoT: bool = False
    celltype_name: Optional[str] = "celltype"
    pos_name: Optional[list[str]] = field(default_factory=lambda: ["x", "y"])
    name_truth: Optional[str] = None
    replicate: str = ""
    openai_url: str = "/v1/chat/completions"
    use_full_name: bool = False
    with_self_type: bool = False  # whether to include self type in the prompt. This won't matter in the new prompt.
    with_region_name: bool = True
    with_domain_name: bool = True
    output_type: str = "niche"
    gpt_model: str = "gpt-4o-mini"
    oneshot_prompt: str = ""
    system_prompt: str = ""
    minimal_gene_threshold: float = 0.0
    prototype_p: float = 0.0
    confident_output: bool = False
    run_root: Optional[str] = None
    def __post_init__(self):
        if self.prototype_p > 0:
            self.model_type = f"{self.model_type}{self.prototype_p}"
        self.refresh_paths()
        self.region1 = f" All cells are from the {self.tissue_region}." if self.with_region_name else ""
        self.region2 = f" in {self.tissue_region}" if self.with_region_name else ""
        self.CoT = "" if not self.with_CoT else "Your CoT prompt here"
        self.text_frequencies = " The frequencies of cell types are also provided." if self.with_numbers else ""
        self.text_expression_numbers = " The expression levels of marker genes are also provided." if self.with_numbers else ""

    def refresh_paths(self) -> None:
        """Refresh generated-artifact paths after changing sample/model settings."""

        resolved_run_root = Path(
            self.run_root or os.environ.get("LLM_ST_RUN_ROOT", ".")
        ).expanduser()
        self.folder_path = str(
            resolved_run_root / "batch_json" / f"{self.data_name}_{self.model_type}"
        )
        self.output_path = str(
            resolved_run_root / "batch_results" / f"{self.data_name}_{self.model_type}"
        )



def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def sig_figs(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))


def check_batch_status(client, batch_id, interval=120):
    while True:
        # 获取当前的批次状态
        status = client.batches.retrieve(batch_id).status

        # 检查是否完成
        if status == "completed":
            print("完成")
            break  # 退出循环
        elif status == "failed":
            print("failed")
            break
        elif status == "cancelled":
            print("cancelled")
            break
        elif status == "cancelling":
            print("cancelling")
            break
        else:
            print(f"当前状态: {status}, 等待 {interval / 60} 分钟后再次检查...")
            print(f"当前进度：{client.batches.retrieve(batch_id).request_counts}")
            time.sleep(interval)
    return(status)



def extract_output_microenvironments(text):
    """
    Extracts the content following 'Outputs:' from the given text.

    Args:
        text (str): The text to search.

    Returns:
        dict: Extracted microenvironment data.
    """
    import re
    import ast

    # Regular expression to match 'Outputs: {...}' or 'Outputs= {...}'
    text = text.replace("\n", "")
    pattern = r'Outputs\s*[:=]\s*({.*?})'
    matches = re.findall(pattern, text)

    extracted_data = {}
    if len(matches) == 0:
        text = text.replace("{", "")
        text = text.replace("}", "")
        print(f"Warning: No matches found for Outputs:{{}}, returning the whole {text}")
        extracted_data['content'] = text
        return extracted_data

    for match in matches:
        # Remove 'Cell ID:' if present
        match = match.replace("Cell ID:", "")
        # Clean up any '{id}' keys to 'id'
        match = re.sub(r"'{(\d+)}'", r"'\1'", match)
        try:
            # Attempt to parse the matched content as a Python literal
            parsed_content = ast.literal_eval(match)
            if isinstance(parsed_content, dict):
                extracted_data.update(parsed_content)
            elif isinstance(parsed_content, (set, list)):
                extracted_data['content'] = list(parsed_content)
            else:
                extracted_data['content'] = parsed_content
        except (SyntaxError, ValueError):
            # If parsing fails, extract content inside braces directly
            inner_match = re.search(r'\{(.*?)\}', match)
            if inner_match:
                content = inner_match.group(1).strip('"')
                extracted_data['content'] = content
            else:
                print(f"Error parsing: {match}")
                continue

    return extracted_data








def sparse_adjacency(locations, threshold, add_diagonal=False):
    """
    Computes a sparse adjacency matrix from location data.

    Args:
        locations (pandas.DataFrame): DataFrame with x, y (or more) coordinate columns.
        threshold (float): Distance threshold for defining neighbors.

    Returns:
        csr_matrix: Sparse adjacency matrix.
        distances: Pairwise distances between all points.
    """
    # Compute pairwise distances
    locations = locations.copy()
    distances = squareform(pdist(locations.values))

    # Apply threshold to find neighbors
    rows, cols = np.where((distances <= threshold) & (distances != 0))

    # Construct sparse adjacency matrix
    data = np.ones(len(rows))
    adj_matrix_sparse = csr_matrix((data, (rows, cols)), shape=distances.shape)

    if add_diagonal:
        # add diagonal to the adj_matrix
        adj_matrix_sparse = adj_matrix_sparse + csr_matrix(np.eye(adj_matrix_sparse.shape[0]))

    return adj_matrix_sparse, distances


def prepare_neighbor_data(adata, config, representetive_gene_list=None):
    """
    Prepare neighbor data for spatial analysis.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing spatial transcriptomics data
    config : object
        Configuration object with attributes: pos_name, celltype_name, r, Graph_type
    representetive_gene_list : list, optional
        List of representative genes for analysis. If None and Graph_type != "count",
        all available genes will be used.

    Returns:
    --------
    neighbor_normalized_df : pd.DataFrame
        Normalized neighbor cell type counts
    neighbor_normalized_df_genes : pd.DataFrame or None
        Normalized neighbor gene expression (if Graph_type != "count")
    adj_matrix : scipy.sparse matrix
        Adjacency matrix with diagonal added
    """
    import scanpy as sc

    pos_data = adata.obs[config.pos_name]
    names_celltypes = adata.obs[config.celltype_name].unique()
    celltype_data = adata.obs[names_celltypes]

    # --- Compute adjacency matrix ---
    r = config.r
    adj_matrix, distances = sparse_adjacency(pos_data, threshold=r)
    # add diagonal to the adj_matrix
    adj_matrix = adj_matrix + scipy.sparse.diags(np.ones(adj_matrix.shape[0]))

    # --- Calculate neighbor counts ---
    neighbor_count = adj_matrix.dot(celltype_data)
    n_neighbors = adj_matrix.sum(axis=1).A1  # .A1 converts to 1D numpy array
    # Convert n_neighbors to a column vector for element-wise division
    n_neighbors_col = n_neighbors.reshape(-1, 1)
    # Perform element-wise division between neighbor_count and n_neighbors_col
    neighbor_matrix_normalized = neighbor_count / n_neighbors_col

    neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized,
                                  index=celltype_data.index,
                                  columns=names_celltypes)

    neighbor_normalized_df_genes = None
    if config.Graph_type != "count":
        # Normalize data
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        # Use provided gene list or all available genes
        if representetive_gene_list is not None:
            top_genes = representetive_gene_list
            common_genes = list(set(adata.var_names) & set(top_genes))
            adata = adata[:, common_genes]
        else:
            # Use all available genes
            common_genes = adata.var_names.tolist()

        # --- Calculate neighbor genes ---
        neighbor_genes = adj_matrix.dot(adata.X)
        # Perform element-wise division between neighbor_count and n_neighbors_col
        neighbor_matrix_normalized_genes = neighbor_genes / n_neighbors_col
        neighbor_normalized_df_genes = pd.DataFrame(neighbor_matrix_normalized_genes,
                                    index=adata.obs_names,
                                    columns=common_genes)

    return neighbor_normalized_df, neighbor_normalized_df_genes, adj_matrix


def run_gemini(model, gen_config, df, config, prompt_func,  n_rows=1, start_idx=0, end_idx=None, df_extra = None, column_name = None):
    df = df.copy()
    gemini_results_df = pd.DataFrame()
    store_responses = []
    if end_idx is None:
        end_idx = len(df)
    for i in range(start_idx, end_idx, n_rows):
        #
        rows = [i for i in range(len(df))[i:i+n_rows]]
        if df_extra is None:
            prompt_q = config.oneshot_prompt + prompt_func(df, rows, config)
        else:
            prompt_q = config.oneshot_prompt + prompt_func(df, df_extra, rows, config)
        response = model.generate_content(prompt_q, generation_config=gen_config)
        store_responses.append(response.text)
        if config.confident_output:
            extract_dict = extract_microenvironments_and_confidence(response.text)
            gemini_results_df = pd.concat([gemini_results_df,
                                           pd.DataFrame({
                                               column_name: extract_dict.environments,
                                               'confidence': extract_dict.confidence
                                           }, index=rows)], ignore_index=False)
        else:
            extract_dict = extract_output_microenvironments(response.text)
            gemini_results_df = pd.concat([gemini_results_df, pd.DataFrame(extract_dict.values(), index=rows)], ignore_index=False)
        # if len(extract_dict) != n_rows:
        #     print(f"Error: {len(extract_dict)} != {n_rows}")
        #     print(f"{i = }")
        #     print(response.text)
        time.sleep(0.2)
        # every 100 rows, save the response
        if (i + 1) % 100 == 0:
            print(i+1)
            with open(f'./gemini_results/{config.data_name}_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.pkl', 'wb') as file:
                pickle.dump(store_responses, file)
            gemini_results_df.to_csv(f"./gemini_results/{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

    if column_name is not None:
        if len(gemini_results_df.columns) == 1:
            gemini_results_df.columns = [column_name]
    return gemini_results_df, store_responses



def _process_deepseek_batch(client, df, config, prompt_func, rows, df_extra, logprobs, max_tokens, column_name, top_logprobs):
    """
    Worker function to process a single batch of rows for DeepSeek API.

    Args:
        client: OpenAI client
        df: DataFrame containing neighbor counts or spatial data
        config: Configuration object with model settings
        prompt_func: Function to generate prompts for each cell
        rows: List of row indices to process
        df_extra: Optional extra DataFrame
        logprobs: Whether to return logprobs
        max_tokens: Maximum number of tokens to generate
        column_name: Name for output column
        top_logprobs: Number of top logprobs to return

    Returns:
        tuple: (batch_results_df, response_content, rows, success)
    """
    try:
        # Generate prompt for this batch
        if df_extra is None:
            prompt_q = config.oneshot_prompt + prompt_func(df, rows, config)
        else:
            prompt_q = config.oneshot_prompt + prompt_func(df, df_extra, rows, config)

        # Prepare messages for DeepSeek API
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt_q}
        ]

        # Make API call to DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek's chat model
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            response_format={'type': 'json_object'}
        )

        # Extract response content
        content = response.choices[0].message.content

        # Process response based on confidence output setting
        if config.confident_output:
            extract_dict = extract_microenvironments_and_confidence(content)
            batch_results_df = pd.DataFrame({
                column_name: extract_dict.environments,
                'confidence': extract_dict.confidence
            }, index=df.index[rows])
        else:
            extract_dict = extract_json_microenvironment(content)
            batch_results_df = pd.DataFrame(extract_dict.values(), index=df.index[rows])

        return batch_results_df, content, rows, True

    except Exception as e:
        print(f"Error processing rows {rows}: {e}")
        # Create error results for failed rows
        if config.confident_output:
            batch_results_df = pd.DataFrame({
                column_name: "Error",
                'confidence': 0.0
            }, index=rows)
        else:
            batch_results_df = pd.DataFrame({"content": "Error"}, index=rows)

        return batch_results_df, f"Error: {e}", rows, False


def run_deepseek(client, df, config, prompt_func, n_rows=1, start_idx=0, end_idx=None, df_extra=None,
                 logprobs=False, max_tokens=1024, column_name=None, max_workers=5):
    """
    Run DeepSeek API for microenvironment prediction on spatial transcriptomics data using parallel processing.

    Args:
        client: OpenAI client
        df: DataFrame containing neighbor counts or spatial data
        config: Configuration object with model settings
        prompt_func: Function to generate prompts for each cell
        n_rows: Number of rows to process per API call (default: 1)
        start_idx: Starting index for processing (default: 0)
        end_idx: Ending index for processing (default: None - process all)
        df_extra: Optional extra DataFrame (e.g., for gene expression data)
        column_name: Name for output column (default: None)
        logprobs: Whether to return logprobs (default: False)
        max_tokens: Maximum number of tokens to generate (default: 1024)
        max_workers: Maximum number of parallel workers (default: 5)
    Returns:
        tuple: (deepseek_results_df, store_responses)
            - deepseek_results_df: DataFrame with predictions
            - store_responses: List of raw API responses
    """

    df = df.copy()
    deepseek_results_df = pd.DataFrame()
    store_responses = []

    if logprobs:
        top_logprobs = 5
    else:
        top_logprobs = None

    if end_idx is None:
        end_idx = len(df)

    # Create batches for parallel processing
    batches = []
    for i in range(start_idx, end_idx, n_rows):
        rows = [j for j in range(len(df))[i:i+n_rows]]
        batches.append(rows)

    print(f"Processing {len(batches)} batches with {max_workers} parallel workers")

    # Thread-safe data structures for collecting results
    results_lock = threading.Lock()
    processed_count = 0

    def update_progress():
        nonlocal processed_count
        with results_lock:
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} batches")

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(
                _process_deepseek_batch,
                client, df, config, prompt_func, rows, df_extra,
                logprobs, max_tokens, column_name, top_logprobs
            ): rows for rows in batches
        }

        # Collect results as they complete
        batch_results = []
        for future in as_completed(future_to_batch):
            rows = future_to_batch[future]
            try:
                batch_df, response_content, batch_rows, success = future.result()
                batch_results.append((batch_rows[0], batch_df, response_content))  # Use first row index for sorting
                update_progress()

                # Optional: Add small delay to respect rate limits
                time.sleep(0.05)  # Reduced delay since we're processing in parallel

            except Exception as e:
                print(f"Batch with rows {rows} generated an exception: {e}")
                # Create error result for this batch
                if config.confident_output:
                    error_df = pd.DataFrame({
                        column_name: "Error",
                        'confidence': 0.0
                    }, index=rows)
                else:
                    error_df = pd.DataFrame({"content": "Error"}, index=rows)

                batch_results.append((rows[0], error_df, f"Error: {e}"))
                update_progress()

    # Sort results by original order and combine
    batch_results.sort(key=lambda x: x[0])  # Sort by first row index

    all_batch_dfs = []
    for _, batch_df, response_content in batch_results:
        all_batch_dfs.append(batch_df)
        store_responses.append(response_content)

    # Combine all batch results
    if all_batch_dfs:
        deepseek_results_df = pd.concat(all_batch_dfs, ignore_index=False)


    output_dir = './deepseek_results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save responses and results
    print(f"Saving results to {output_dir}{config.data_name}_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}")
    with open(f'{output_dir}{config.data_name}_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.pkl', 'wb') as file:
        pickle.dump(store_responses, file)
    deepseek_results_df.to_csv(f"{output_dir}{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")

    # Set column name if specified
    if column_name is not None:
        if len(deepseek_results_df.columns) == 1:
            deepseek_results_df.columns = [column_name]

    print(f"Completed processing {len(batch_results)} batches")
    return deepseek_results_df, store_responses




def generate_json_end2end(df,
                          config,
                          prompt_func,
                          batch_size = 5000,
                          max_completion_tokens = 1024,
                          n_rows = 1,
                          df_extra = None,
                          logprobs = True,
                          response_format_style = "openai"):
    """
    df: data frame of neighbor counts
    config: config of the experiment
    prompt_func: the function to generate prompt
    batch_size: number of rows in each batch
    n_rows: number of cells to be processed in one prompt
    df_extra: data frame of neighbor genes (optional, if use countPlusGene)
    """
    df = df.copy()
    num_batches = int(((len(df)/n_rows) // batch_size) + 1)  # 计算总的批次数
    start_idx = 0
    end_idx = 0

    # 遍历每个批次
    for batch_num in range(num_batches):
        # 计算当前批次的起始和结束行
        start_idx = end_idx
        end_idx = min(start_idx + batch_size * n_rows, len(df))

        # 生成当前批次的文件名
        output_folder = f"batch_json/{config.data_name}_{config.model_type}/"
        if not os.path.exists(output_folder):
            # Create the folder if it doesn't exist
            os.makedirs(output_folder)
        output_file = f"{output_folder}{config.data_name}_{batch_num + 1}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.json"
        print(f"Generating json from row {start_idx} to row {end_idx} into {output_file}")
        # 打开文件用于写入
        with open(output_file, 'w') as f:
            enum_values = list(config.domain_mapping.values())
            if response_format_style == "openai":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "microenvironment_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "microenvironment": {
                                    "type": "string",
                                    "enum": enum_values
                                }
                            },
                            "required": ["microenvironment"],
                            "additionalProperties": False
                        }
                    }
                }
            else:
                print(f"response_format_style is not supported {response_format_style}. ")
                print("set response_format to None")   #  because only openai support batch request
                response_format = None

            for i in range(start_idx, end_idx, n_rows):
                rows = [i for i in range(len(df))[i:i+n_rows]]
                if df_extra is None:
                    p = config.oneshot_prompt + prompt_func(df, rows, config)
                else:
                    p = config.oneshot_prompt + prompt_func(df, df_extra, rows, config)
                custom_id = df.index[i] if n_rows == 1 else i

                row_data = {
                    "custom_id": str(custom_id),
                    "method": "POST",
                    "url": config.openai_url,
                    "body": {
                        "model": config.gpt_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": config.system_prompt
                            },
                            {
                                "role": "user",
                                "content": p
                            }
                        ],
                        "max_completion_tokens": max_completion_tokens,
                        "logprobs": logprobs,
                        "response_format": response_format
                    }
                }
                # 将字典转为JSON字符串，并写入文件
                json_str = json.dumps(row_data)
                f.write(json_str + '\n')  # 每个JSON对象占一行




# calculate the median distance of neighbors for each cell based on their type
# --- Calculate neighbor distances ---
def calculate_neighbor_distances(adj_matrix, distances, one_hot_matrix):
    """
    Calculates the median distance of neighbors for each cell type.

    Args:
        adj_matrix (csr_matrix): Sparse adjacency matrix.
        distances (np.ndarray): Pairwise distances between all cells.
        one_hot_matrix (csr_matrix): One-hot encoded cell type matrix.

    Returns:
        pd.DataFrame: DataFrame containing median neighbor distances per cell type.
    """
    adj_matrix = adj_matrix.copy()
    distances = distances.copy()
    one_hot_matrix = one_hot_matrix.copy()
    num_cells = adj_matrix.shape[0]
    cell_types = one_hot_matrix.shape[1]  # Number of different cell types
    median_distances = np.full((num_cells, cell_types), np.nan)  # Initialize all as NaN

    # Loop over each cell
    for i in range(num_cells):
        # Find neighbors of cell `i`
        neighbors = adj_matrix[i].nonzero()[1]  # Indices of neighbors

        # If no neighbors, skip to the next cell
        if len(neighbors) == 0:
            continue

        # Loop over each cell type
        for cell_type_idx in range(cell_types):
            # Find neighbors of this type (non-zero values in the one-hot matrix)
            type_neighbors = neighbors[one_hot_matrix[neighbors, cell_type_idx].toarray().flatten() > 0]

            # If no neighbors of this type, skip
            if len(type_neighbors) == 0:
                median_distances[i, cell_type_idx] = np.nan
            else:
                # Compute median distance of neighbors of this type
                median_distances[i, cell_type_idx] = np.median(distances[i, type_neighbors])

    # Convert to DataFrame for easier access
    median_distances_df = pd.DataFrame(median_distances)

    return median_distances_df



def get_farthest_cell_types(
    cell_coordinates: pd.DataFrame,
    cell_types: pd.DataFrame,
    target_cell_id: str,
    config: Config,
    n: int = 3
) -> list:
    """
    Get the top 2 farthest cell types for a given cell.

    Args:
        cell_coordinates: DataFrame with columns 'cell_id', 'x', 'y'
                          representing the spatial coordinates of cells.
        cell_types: DataFrame with columns 'cell_id', 'cell_type'.
        target_cell_id: The ID of the cell to find the farthest cell types from.

    Returns:
        A list of the top 2 farthest cell types, sorted by distance in descending order.
    """
    # check cell_id
    cell_coordinates = cell_coordinates.copy()
    cell_types = cell_types.copy()
    target_cell_id = cell_coordinates.index[target_cell_id]
    cell_coordinates['cell_id'] = cell_coordinates.index
    cell_types['cell_id'] = cell_types.index
    # Merge dataframes on 'cell_id'
    merged_df = pd.merge(cell_coordinates, cell_types, on='cell_id')

    # Get coordinates of the target cell
    target_cell = merged_df[merged_df['cell_id'] == target_cell_id]
    target_x, target_y = target_cell['x'].values[0], target_cell['y'].values[0]

    # Calculate distances between the target cell and all other cells
    merged_df['distance'] = cdist(
        [(target_x, target_y)], merged_df[['x', 'y']]
    ).flatten()

    # Group by cell type and get the average distance
    avg_distances = merged_df.groupby('cell_type', observed=False)['distance'].median()

    # Get the top 2 farthest cell types
    farthest_types = avg_distances.nlargest(n).index.tolist()

    # full cell name
    if config.use_full_name:
        farthest_types = [config.cell_names_mapping.get(name) for name in farthest_types]

    return farthest_types



#########################
# discarded functions
#########################

def extract_last_braces(text):
    # 使用正则表达式找到最后一个花括号中的内容
    matches = re.findall(r'\{[^{}]*\}', text)
    if matches:
        # 提取最后一个匹配到的花括号内容
        last_braces_content = matches[-1]
        try:
            # 将花括号内的内容转换为字典
            braces_dict = ast.literal_eval(last_braces_content)
            if isinstance(braces_dict, dict):
                return braces_dict
            else:
                raise ValueError("内容不是有效的字典格式")
        except (SyntaxError, ValueError) as e:
            return f"错误：{e}"
    else:
        return "未找到花括号内容"


def relabel_cells(adj_matrix, labels):
    """
    Relabels cells based on the majority label of their neighbors.

    Args:
        adj_matrix: A NumPy array representing the adjacency matrix. for example, val_adj_matrix.toarray()
        labels: A Pandas Series containing the labels for each cell. for example, gpt_results_df['gpt4o_mini']

    Returns:
        A Pandas Series containing the updated labels.
    """

    n_cells = len(labels)
    updated_labels = labels.copy()  # Create a copy to avoid modifying the original

    for i in range(n_cells):
        neighbors = np.where(adj_matrix[i] == 1)[0]  # Find indices of neighbors

        if neighbors.size > 0:  # Check if the cell has neighbors
            neighbor_labels = labels.iloc[neighbors]

            # Count occurrences of each label among neighbors
            label_counts = neighbor_labels.value_counts()

            # Get the most frequent label among neighbors
            majority_label = label_counts.index[0]

            # Calculate the proportion of neighbors with the majority label
            majority_proportion = label_counts.iloc[0] / len(neighbors)

            # Relabel if the majority label is different and represents more than half of the neighbors
            if majority_label != updated_labels.iloc[i] and majority_proportion > 0.5:
                updated_labels.iloc[i] = majority_label

    return updated_labels


from itertools import combinations

# Example data structure
# Let's assume your data looks something like this:
# cell_id | method1 | method2 | method3
#    1    |    A    |    X    |    P
#    2    |    A    |    X    |    P
#    3    |    B    |    Y    |    Q

def find_conserved_groups(clustering_df, min_methods=2, min_group_size=2):
    """
    Find groups of cells that consistently cluster together across different methods.

    Parameters:
    -----------
    clustering_df : pandas DataFrame
        DataFrame where rows are cells and columns are different clustering methods
    min_methods : int
        Minimum number of methods that must agree
    min_group_size : int
        Minimum number of cells in a conserved group

    Returns:
    --------
    dict : Dictionary of conserved groups with their occurrence patterns
    """

    # Get all method combinations to check
    methods = clustering_df.columns
    all_combinations = []
    for r in range(min_methods, len(methods) + 1):
        all_combinations.extend(combinations(methods, r))

    conserved_groups = {}

    # Check each combination of methods
    for method_combo in all_combinations:
        # Get subset of data for current methods
        subset = clustering_df[list(method_combo)]

        # Group cells by their cluster assignments
        grouped = subset.groupby(list(method_combo)).apply(lambda x: set(x.index))

        # Filter for groups meeting minimum size requirement
        valid_groups = grouped[grouped.apply(len) >= min_group_size]

        # Store valid groups
        for cell_group in valid_groups:
            group_key = frozenset(cell_group)
            if group_key not in conserved_groups:
                conserved_groups[group_key] = {
                    'cells': sorted(cell_group),
                    'methods': [method_combo],
                    'num_methods': len(method_combo)
                }
            else:
                conserved_groups[group_key]['methods'].append(method_combo)
                conserved_groups[group_key]['num_methods'] = max(conserved_groups[group_key]['num_methods'], len(method_combo))

    return conserved_groups


def create_group_labels_df(conserved_groups, all_cell_indices):
    """
    Convert conserved_groups output into a DataFrame with group labels.

    Parameters:
    -----------
    conserved_groups : dict
        Output from find_conserved_groups function
    all_cell_indices : array-like
        List/array of all cell indices to ensure complete coverage

    Returns:
    --------
    pd.DataFrame
        DataFrame with one column containing group labels
    """
    # Initialize DataFrame with "unassigned" as default
    group_labels = pd.DataFrame(
        "unassigned",
        index=all_cell_indices,
        columns=['conserved_label']
    )

    # Assign group numbers to cells
    for group_num, (cells, info) in enumerate(conserved_groups.items(), 1):
        index_list = [str(cell) for cell in info['cells']]
        group_labels.loc[index_list, 'conserved_label'] = f'group_{group_num}'

    return group_labels



# use top k nearest neighbors of one cell to find the conserved cells
def find_high_confidence_cells(adata, label_key, pos_data = None,  k=10, distance_threshold=100):
    """
    Find high confidence cells based on neighbor label consistency.

    Args:
        adata: AnnData object containing spatial coordinates
        label_key: Key for the label vector in adata.obs
        k: Number of nearest neighbors to consider
        distance_threshold: Maximum distance threshold for neighbors

    Returns:
        high_confidence_mask: Boolean mask indicating high confidence cells
    """
    if pos_data is None:
        pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)

    # Compute pairwise distances
    distances = squareform(pdist(pos_data.values))

    # Initialize mask for high confidence cells
    high_confidence_mask = np.zeros(len(adata), dtype=bool)

    # For each cell
    for i in range(len(adata)):
        # Get distances to other cells
        cell_distances = distances[i]

        # Find indices of k nearest neighbors within distance threshold
        valid_neighbors = np.where(cell_distances <= distance_threshold)[0]
        if len(valid_neighbors) > k:
            # Get k nearest among valid neighbors
            valid_neighbors = valid_neighbors[np.argsort(cell_distances[valid_neighbors])[:k]]

        if len(valid_neighbors) > 0:
            # Get cell's label and neighbor labels
            cell_label = adata.obs[label_key].iloc[i]
            neighbor_labels = adata.obs[label_key].iloc[valid_neighbors]

            # Check if all neighbors have same label as cell
            if all(neighbor_labels == cell_label):
                high_confidence_mask[i] = True

    return high_confidence_mask

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MicroenvironmentResult:
    """Class for storing a microenvironment and its confidence score."""
    environments: str
    confidence: float

def extract_microenvironments_and_confidence(text: str) -> MicroenvironmentResult:
    """
    Extracts a microenvironment and its confidence score from the given text.

    Args:
        text (str): The text containing microenvironment and confidence score.
                   Expected format like "['Layer 1': 0.8]" or similar.

    Returns:
        MicroenvironmentResult: An object with environment (str) and confidence (float) attributes.
    """
    import re
    import ast
    import json

    # Replace newlines for easier pattern matching
    text = text.replace("\n", " ")

    # Try to find patterns like ['key': value] or {'key': value}
    pattern = r'[\[\{][\'\"](.+?)[\'\"]:\s*([0-9.]+)[\]\}]'
    matches = re.findall(pattern, text)

    # Default values
    environment = ""
    confidence_score = 0.0

    if not matches:
        # Try to find dictionary-like patterns and convert to valid Python syntax
        try:
            # Replace single quotes with double quotes for keys
            processed_text = re.sub(r"(\[|\{)\s*'([^']+)':", r'{\"\2\":', text)
            processed_text = processed_text.replace("]", "}")

            # Try to find dictionary pattern in the processed text
            dict_pattern = r'({.*?})'
            dict_matches = re.findall(dict_pattern, processed_text)

            if dict_matches:
                for match in dict_matches:
                    try:
                        # Try to parse as Python dictionary
                        parsed_dict = ast.literal_eval(match)
                        if isinstance(parsed_dict, dict) and len(parsed_dict) > 0:
                            # Get the first key-value pair
                            env, conf = next(iter(parsed_dict.items()))
                            environment = env
                            if isinstance(conf, str):
                                conf = float(conf)
                            confidence_score = conf
                            break  # Only take the first key-value pair
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing processed dictionary: {e}")
                        continue
            else:
                print(f"Warning: No dictionary-like patterns found in the text: {text[:100]}...")
        except Exception as e:
            print(f"Error processing text: {e}")
            print(f"Original text: {text[:100]}...")
    else:
        # Process direct regex matches - take only the first match
        environment, conf = matches[0]
        try:
            confidence_score = float(conf)
        except ValueError:
            print(f"Warning: Could not convert confidence '{conf}' to float")
            confidence_score = 0.0

    return MicroenvironmentResult(environments=environment, confidence=confidence_score)


def calculate_neighborhood_ari(adata_obs, adj_matrix, name_1, name_2):
    """
    Calculate ARI between two clustering labels for each cell's neighborhood

    Parameters:
    -----------
    adata_obs : pd.DataFrame
        DataFrame containing 'niche' and 'zeroshot_gpt4o_mini_refined' columns
    adj_matrix : np.ndarray or sparse matrix
        Adjacency matrix of cells

    Returns:
    --------
    pd.Series
        ARI scores for each cell's neighborhood
    """
    # Convert adjacency matrix to sparse if it isn't already
    if not scipy.sparse.issparse(adj_matrix):
        adj_matrix = scipy.sparse.csr_matrix(adj_matrix)

    # Initialize array to store ARI scores
    neighborhood_ari = np.zeros(len(adata_obs))

    # Get labels as numpy arrays
    niche_labels = adata_obs[name_1].values
    zeroshot_labels = adata_obs[name_2].values

    # Calculate ARI for each cell's neighborhood
    for i in range(len(adata_obs)):
        # Get indices of neighboring cells (including self)
        neighbors = adj_matrix[i].nonzero()[1]

        if len(neighbors) > 1:  # Only calculate if there are neighbors
            # Get labels for the neighborhood
            neighborhood_niche = niche_labels[neighbors]
            neighborhood_zeroshot = zeroshot_labels[neighbors]

            # Calculate ARI for the neighborhood
            ari = adjusted_rand_score(neighborhood_niche, neighborhood_zeroshot)
            neighborhood_ari[i] = ari
        else:
            neighborhood_ari[i] = np.nan  # Set to NaN if no neighbors

    # Return as pandas dataframe
    return pd.DataFrame(
        neighborhood_ari,
        index=adata_obs.index,
        columns=['neighborhood_ari']
    )




# Step 1: Run the function for all rows and store results
def get_unique_prompts(df, config, prompt_func, df_extra=None):
    """
    Identify rows that produce the same prompt text when passed through the zeroshot_celltype function.

    Parameters:
    -----------
    df : pandas.DataFrame
        The neighbor_normalized_df dataframe
    config : object
        The configuration object with parameters for zeroshot_celltype
    prompt_func : function
        The prompt function to use
    df_extra : pandas.DataFrame, optional
        Additional dataframe (e.g., for gene expression data)
    Returns:
    --------
    tuple: (unique_indices, idx_mapping, inverse_mapping)
        - unique_indices: indices of only unique rows
        - idx_mapping: Dictionary mapping original indices to unique indices
        - inverse_mapping: Dictionary mapping unique indices to lists of original indices
    """
    all_prompts = {}

    # Generate prompts for each row and track duplicates
    print("Generating prompts for each row...")
    for i in tqdm(range(len(df))):
        if config.Graph_type != "countPlusGenes":
            prompt_text = prompt_func(df, [i], config)
        else:
            prompt_text = prompt_func(df, df_extra, [i], config)

        # Use the prompt text as a key to find duplicates
        if prompt_text not in all_prompts:
            all_prompts[prompt_text] = []
        all_prompts[prompt_text].append(i)

    # Create a mapping from original indices to unique indices
    idx_mapping = {}
    inverse_mapping = {}
    unique_indices = []

    for unique_idx, (_, indices) in enumerate(all_prompts.items()):
        representative_idx = indices[0]  # Use the first index as representative
        unique_indices.append(representative_idx)
        inverse_mapping[unique_idx] = indices

        for idx in indices:
            idx_mapping[idx] = unique_idx


    print(f"Reduced from {len(df)} rows to {len(unique_indices)} unique prompts")

    return unique_indices, idx_mapping, inverse_mapping

# Step 2: Use the results to map back to full labels
def restore_full_labels_df(reduced_labels_df, inverse_mapping):
    """
    Restore full labels from reduced dataframe results

    Parameters:
    -----------
    reduced_labels_df : pandas.DataFrame
        DataFrame with labels for the reduced/deduplicated dataset
    inverse_mapping : dict
        Mapping from reduced indices to lists of original indices

    Returns:
    --------
    pandas.DataFrame: Full labels corresponding to the original dataset
    """
    # Determine the length of the original dataset
    original_length = max(max(idxs) for idxs in inverse_mapping.values()) + 1

    # Extract column name from the reduced labels dataframe
    column_name = reduced_labels_df.columns[0]

    # Initialize a new dataframe for full labels with the same column name
    full_labels_df = pd.DataFrame(index=range(original_length), columns=[column_name])

    # Map reduced labels back to original indices
    for reduced_idx, orig_indices in inverse_mapping.items():
        label = reduced_labels_df.iloc[reduced_idx, 0]  # Get the label from the first column
        for orig_idx in orig_indices:
            full_labels_df.loc[orig_idx, column_name] = label

    return full_labels_df


def restore_full_data_from_whole_df(whole_df, reduced_labels_df, config, prompt_func, df_extra=None):
    """
    Restore full data labels from whole_df using reduced_labels_df by generating prompts.

    This function generates prompts for each row in whole_df and matches them with prompts
    from reduced_labels_df to assign labels to the full data.

    Parameters:
    -----------
    whole_df : pandas.DataFrame
        The complete dataframe containing all data
    reduced_labels_df : pandas.DataFrame
        DataFrame with predicted labels for the reduced/deduplicated dataset.
        Index corresponds to reduced version of full data, has one column with predicted labels.
    config : object
        The configuration object with parameters for prompt generation
    prompt_func : function
        The prompt function to use for generating prompts
    df_extra : pandas.DataFrame, optional
        Additional dataframe (e.g., for gene expression data)

    Returns:
    --------
    pandas.DataFrame: Labels for the full data with same index as the matching rows in whole_df
    """
    # Generate prompts for reduced indices and create prompt-to-label mapping
    print("Generating reference prompts from reduced_labels_df...")
    prompt_to_label = {}

    for idx in tqdm(reduced_labels_df.index):
        # Convert index to integer position if needed
        if isinstance(idx, (str, object)) and not isinstance(idx, int):
            int_idx = whole_df.index.get_loc(idx)
        else:
            int_idx = idx

        if config.Graph_type != "countPlusGenes":
            prompt_text = prompt_func(whole_df, [int_idx], config)
        else:
            prompt_text = prompt_func(whole_df, df_extra, [int_idx], config)

        # Map prompt to its corresponding label
        label = reduced_labels_df.loc[idx].iloc[0]  # Get the label from the first (and only) column
        prompt_to_label[prompt_text] = label

    # Find all rows in whole_df that match the reference prompts and assign labels
    print("Finding matching rows in whole_df and assigning labels...")
    full_data_labels = []
    full_data_indices = []

    for i in tqdm(range(len(whole_df))):
        if config.Graph_type != "countPlusGenes":
            prompt_text = prompt_func(whole_df, [i], config)
        else:
            prompt_text = prompt_func(whole_df, df_extra, [i], config)

        if prompt_text in prompt_to_label:
            full_data_labels.append(prompt_to_label[prompt_text])
            full_data_indices.append(whole_df.index[i])

    # Create DataFrame with the same column name as reduced_labels_df
    column_name = reduced_labels_df.columns[0]
    full_labels_df = pd.DataFrame(
        {column_name: full_data_labels},
        index=full_data_indices
    )

    print(f"Assigned labels to {len(full_labels_df)} rows from {len(whole_df)} total rows")
    print(f"Original reduced labels: {len(reduced_labels_df)}")

    return full_labels_df


def process_batch_results(config, n_batch=1, extract_function=None):
    """
    Process batch results from GPT API response files and extract data into a DataFrame.

    Parameters:
    -----------
    config : object
        Configuration object containing data_name, output_path, and other parameters
    n_batch : int, default=1
        Number of batches to process
    extract_function : callable, default=None
        Function to extract outputs from content. If None, uses extract_output_microenvironments

    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted results with custom_id as index
    """
    if extract_function is None:
        extract_function = extract_output_microenvironments

    gpt_results_df = pd.DataFrame()

    for i in range(1, n_batch + 1):
        save_name = f"response_{config.data_name}_{i}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.txt"
        output_file_name = f"{config.output_path}/{save_name}"

        try:
            # Read file and process line by line
            with open(output_file_name, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        # Parse JSON string from each line
                        json_data = json.loads(line.strip())

                        # Extract custom_id and content information
                        custom_id = json_data['custom_id']
                        content = json_data['response']['body']['choices'][0]['message']['content']

                        # Clean up content formatting
                        content = (content.replace('\n-', " ")
                                         .replace('```', "")
                                         .replace('json', "")
                                         .replace('plaintext', '')
                                         .replace('python', '')
                                         .replace('Output:', 'Outputs:')
                                         .replace('\n', " "))

                        # Extract outputs using the provided function
                        extract_dict = extract_function(content)

                        # Add extracted information to DataFrame
                        gpt_results_df = pd.concat([
                            gpt_results_df,
                            pd.DataFrame(extract_dict.values(), index=[custom_id])
                        ], ignore_index=False)

                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError in batch {i}, line {line_num}: {e}")
                        print(f"Problematic line: {line[:100]}...")
                    except KeyError as e:
                        print(f"KeyError in batch {i}, line {line_num}: Missing key {e}")
                    except Exception as e:
                        print(f"Unexpected error in batch {i}, line {line_num}: {e}")

        except FileNotFoundError:
            print(f"File not found: {output_file_name}")
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    return gpt_results_df




def create_prediction(df, p, ground_truth_name='ground_truth', random_state=42):
    """
    Create prediction column based on randomly sampling proportion p of data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'cluster' and ground_truth_name columns
    p : float
        Proportion of data to randomly sample (between 0 and 1)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        DataFrame with new 'prediction' column
    """
    # Make a copy to avoid modifying original dataframe
    df_result = df.copy()

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Randomly sample proportion p from each cluster
    sampled_indices = []
    for cluster_id in df['cluster'].unique():
        if pd.isna(cluster_id):
            continue
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_sample_size = int(round(len(cluster_data) * p))
        if cluster_sample_size > 0:
            cluster_sampled_indices = np.random.choice(
                cluster_data.index,
                size=min(cluster_sample_size, len(cluster_data)),
                replace=False
            )
            sampled_indices.extend(cluster_sampled_indices)
        else:
            cluster_sample_size = 1
            cluster_sampled_indices = np.random.choice(
                cluster_data.index,
                size=min(cluster_sample_size, len(cluster_data)),
                replace=False
            )
            sampled_indices.extend(cluster_sampled_indices)


    sampled_df = df.loc[sampled_indices]

    print(f"Original dataset size: {len(df)}")
    print(f"Total sample size (p={p} from each cluster): {len(sampled_indices)}")

    # Group by cluster and find majority ground_truth for each cluster
    cluster_to_prediction = {}

    for cluster_id in sampled_df['cluster'].unique():
        if pd.isna(cluster_id):
            continue
        cluster_data = sampled_df[sampled_df['cluster'] == cluster_id]

        # Count occurrences of each ground_truth in this cluster
        ground_truth_counts = cluster_data[ground_truth_name].value_counts()

        # Get the majority ground_truth (most frequent)
        majority_ground_truth = ground_truth_counts.index[0]

        cluster_to_prediction[cluster_id] = majority_ground_truth

        print(f"Cluster {cluster_id}: {len(cluster_data)} samples, "
              f"majority ground_truth = {majority_ground_truth} "
              f"(count: {ground_truth_counts.iloc[0]}/{len(cluster_data)})")

    # Map all rows' cluster to prediction_BASS based on the mapping
    df_result['prediction'] = df_result['cluster'].map(cluster_to_prediction)

    # Handle any clusters that weren't in the sample (assign NaN or most common prediction)
    missing_clusters = df_result[df_result['prediction'].isna()]['cluster'].unique()
    if len(missing_clusters) > 0:
        print(f"Warning: {len(missing_clusters)} clusters not found in sample: {missing_clusters}")
        # You could assign the most common prediction or leave as NaN
        # For now, we'll leave as NaN

    return df_result


def extract_json_microenvironment(content):
    """
    Extracts microenvironment value from JSON-formatted content.
    Handles cases where content is like '{"microenvironment":"Layer 6"}' or similar JSON structures.

    Args:
        content (str): The content string which might be JSON formatted

    Returns:
        dict: Extracted microenvironment data with extracted value
    """
    import json
    import re

    extracted_data = {}

    # First try to parse as direct JSON
    try:
        # Clean up the content - remove extra whitespace and formatting
        cleaned_content = content.strip()

        # fix a mistake in older version of the code
        if '{microenvironment: \'' in cleaned_content:
            cleaned_content = cleaned_content.replace('{microenvironment: \'', '{\"microenvironment\": \"')
            cleaned_content = cleaned_content.replace('\'}', '\"}')
        # Try to parse as JSON directly
        json_data = json.loads(cleaned_content)

        if isinstance(json_data, dict):
            # Look for microenvironment key
            if 'microenvironment' in json_data:
                extracted_data['content'] = json_data['microenvironment']
                return extracted_data
            # If no microenvironment key, take the first value
            elif json_data:
                first_key = next(iter(json_data))
                extracted_data['content'] = json_data[first_key]
                return extracted_data

    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, try to extract JSON-like patterns
        pass

    # Try to find JSON patterns in the text
    json_pattern = r'\{[^}]*\}'
    matches = re.findall(json_pattern, cleaned_content)

    for match in matches:
        try:
            json_data = json.loads(match)
            if isinstance(json_data, dict):
                if 'microenvironment' in json_data:
                    extracted_data['content'] = json_data['microenvironment']
                    return extracted_data
                elif json_data:
                    first_key = next(iter(json_data))
                    extracted_data['content'] = json_data[first_key]
                    return extracted_data
        except (json.JSONDecodeError, ValueError):
            continue

    # If no JSON found, fall back to the original extraction method
    return extract_output_microenvironments(content)




def plot_metric_comparison(all_results_df, metric, experiment_type,
                          data_names = ['BZ5', 'BZ9', 'BZ14'],
                          model_names = ['Llama8', 'Qwen30', 'Llama70', 'gpt4o_mini'],
                          figsize=(12, 6), palette='Set2', show_values=False, ylim=None):
    """
    Plot barplot with error bars comparing different models for each dataset.

    Parameters:
    -----------
    all_results_df : pd.DataFrame
        DataFrame containing columns: data_name, model_name, experiment_types, and metric columns
    metric : str
        The metric to plot (e.g., 'ARI', 'NMI', 'HOM', etc.)
    experiment_type : str
        The experiment type to filter for (e.g., 'zeroshot', 'finetune_BZ5')
    figsize : tuple, optional
        Figure size (width, height)
    palette : str, optional
        Color palette for different models
    show_values : bool, optional
        Whether to show values on top of bars
    ylim : tuple, optional
        Y-axis limits (min, max)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Filter data for the specified experiment type
    filtered_df = all_results_df[all_results_df['experiment_types'] == experiment_type].copy()

    if filtered_df.empty:
        print(f"No data found for experiment_type: {experiment_type}")
        return None, None

    # Calculate mean and std for each data_name and model_name combination
    summary_stats = filtered_df.groupby(['data_name', 'model_name'])[metric].agg(['mean', 'std', 'count']).reset_index()

    # Get unique data names and model names


    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    x = np.arange(len(data_names))
    width = 0.8 / len(model_names)  # Width of bars

    # Create color palette
    colors = sns.color_palette(palette, len(model_names))

    # Plot bars for each model
    for i, model in enumerate(model_names):
        model_data = summary_stats[summary_stats['model_name'] == model]

        means = []
        stds = []
        for data_name in data_names:
            data_point = model_data[model_data['data_name'] == data_name]
            if not data_point.empty:
                means.append(data_point['mean'].values[0])
                stds.append(data_point['std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        # Plot bars with error bars
        positions = x + (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(positions, means, width, yerr=stds,
                     label=model, color=colors[i], alpha=0.8,
                     capsize=5, error_kw={'linewidth': 1.5})

        # Optionally show values on top of bars
        if show_values:
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                if mean > 0:  # Only show if there's data
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=0)

    # Customize plot
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison Across Models\n({experiment_type})',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(data_names, rotation=45, ha='right')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    return fig, ax




def plot_boxplot_comparison_by_dataset(all_results_df, metric,
                                       data_names=['BZ5', 'BZ9', 'BZ14'],
                                       baseline_models=None, compare_models=None,
                                       baseline_exp='zeroshot', compare_exp='finetune_BZ5',
                                       figsize=(15, 5), palette=None, ylim=None, show_points=True):
    """
    Plot boxplot comparing two experiment types (baseline vs comparison) across different models with pair-to-pair comparison.
    Creates separate subplots for each dataset (data_name).

    Parameters:
    -----------
    all_results_df : pd.DataFrame
        DataFrame containing columns: data_name, model_name, experiment_types, and metric columns
    metric : str
        The metric to plot (e.g., 'ARI', 'NMI', 'HOM', etc.)
    data_names : list, optional
        List of data names to plot.
        If None, uses default: ['BZ5', 'BZ9', 'BZ14']
    baseline_models : list, optional
        List of model names to use for baseline experiment.
        If None, uses default: ['Llama8', 'Qwen30', 'Llama70', 'gpt4o_mini']
    compare_models : list, optional
        List of model names to use for comparison experiment. Should have same length as baseline_models.
        If None, uses the same models as baseline_models.
    baseline_exp : str, optional
        The baseline experiment type name in the dataframe (default: 'zeroshot')
    compare_exp : str, optional
        The comparison experiment type name in the dataframe (default: 'finetune_BZ5')
    figsize : tuple, optional
        Figure size (width, height)
    palette : dict or str, optional
        Color palette for experiment types. If dict, keys should be experiment type names (values of baseline_exp and compare_exp).
    ylim : tuple, optional
        Y-axis limits (min, max)
    show_points : bool, optional
        Whether to show individual data points on top of boxes

    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    # Set default model lists if not provided
    if baseline_models is None:
        baseline_models = ['Llama8', 'Qwen30', 'Llama70', 'gpt4o_mini']

    if compare_models is None:
        compare_models = baseline_models.copy()

    # Validate that both lists have the same length
    if len(baseline_models) != len(compare_models):
        raise ValueError(f"baseline_models and compare_models must have the same length. "
                        f"Got {len(baseline_models)} and {len(compare_models)} respectively.")

    # Filter data for the two experiment types
    filtered_df = all_results_df[
        all_results_df['experiment_types'].isin([baseline_exp, compare_exp])
    ].copy()

    if filtered_df.empty:
        print(f"No data found for experiment types: {baseline_exp} or {compare_exp}")
        return None, None

    # Get unique datasets
    n_datasets = len(data_names)

    # Create subplots
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=True)
    if n_datasets == 1:
        axes = [axes]

    # Define colors
    if palette is None:
        palette = {
            baseline_exp: '#8dd3c7',
            compare_exp: '#fb8072'
        }

    # Plot each dataset
    for idx, data_name in enumerate(data_names):
        ax = axes[idx]

        # Filter for this dataset
        dataset_df = filtered_df[filtered_df['data_name'] == data_name].copy()

        # Check if data exists for this dataset
        if dataset_df.empty:
            continue

        # Create a combined column for x-axis grouping
        dataset_df['model_exp'] = dataset_df['model_name'] + '\n' + dataset_df['experiment_types']

        # Define the order for x-axis (alternating baseline and comparison for each model pair)
        x_order = []
        for b_model, c_model in zip(baseline_models, compare_models):
            x_order.append(f"{b_model}\n{baseline_exp}")
            x_order.append(f"{c_model}\n{compare_exp}")

        # Filter to only include models that exist in the data
        x_order = [x for x in x_order if x in dataset_df['model_exp'].values]

        # Create color list for each box based on experiment type
        colors = []
        for x in x_order:
            if baseline_exp in x:
                colors.append(palette.get(baseline_exp, '#8dd3c7'))
            else:
                colors.append(palette.get(compare_exp, '#fb8072'))

        if not x_order:
            continue

        # Create boxplot
        box_data = [dataset_df[dataset_df['model_exp'] == x][metric].values for x in x_order]

        bp = ax.boxplot(box_data,
                        labels=x_order,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=6,
                                      markeredgecolor='darkred', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Optionally add individual points
        if show_points:
            for i, x in enumerate(x_order):
                data = dataset_df[dataset_df['model_exp'] == x][metric].values
                # Add jitter to x-coordinates
                x_coords = np.random.normal(i + 1, 0.04, size=len(data))
                ax.scatter(x_coords, data, alpha=0.4, s=30, color='black', zorder=3)

        # Customize x-axis labels to show model names more clearly
        x_labels = []
        for label in x_order:
            model = label.split('\n')[0]
            exp_type_val = label.split('\n')[1]

            # Create short labels
            if exp_type_val == 'zeroshot':
                short_exp = 'ZS'
            elif 'finetune' in exp_type_val:
                short_exp = 'FT'
            elif exp_type_val == 'ttrlgc':
                short_exp = 'TTRL-GC'
            else:
                # Fallback: take first 4 chars uppercase or just use the name
                short_exp = exp_type_val[:4].upper() if len(exp_type_val) > 4 else exp_type_val

            x_labels.append(f"{model}\n({short_exp})")

        ax.set_xticklabels(x_labels, fontsize=9)

        # Add vertical lines to separate models
        for i in range(2, len(x_order), 2):
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # Customize subplot
        ax.set_title(f'{data_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set y-axis limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # Only add y-label to leftmost subplot
        if idx == 0:
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')

    # Create custom legend on the rightmost subplot
    legend_elements = [
        Patch(facecolor=palette.get(baseline_exp, '#8dd3c7'), alpha=0.7, label=baseline_exp),
        Patch(facecolor=palette.get(compare_exp, '#fb8072'), alpha=0.7, label=compare_exp),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   markersize=8, markeredgecolor='darkred', label='Mean')
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Add overall title
    fig.suptitle(f'{metric} Comparison: {baseline_exp} vs {compare_exp} by Dataset',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    return fig, axes
