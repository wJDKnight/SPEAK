import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

from src.paths import data_root, dataset_root, run_root

########################################################
# parameters
########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--prepare_ref_data', action='store_true')
parser.add_argument('--downsample', action='store_true')
parser.add_argument(
    '--processed_dir',
    type=str,
    default=os.environ.get(
        'LLM_ST_DECONV_REFERENCE_ROOT',
        str(data_root() / 'processed_inputs' / 'spatialDLPFC_new'),
    ),
)
parser.add_argument('--prepare_spatial_data', action='store_true')
parser.add_argument('--sample_id', type=str, default='151509')
parser.add_argument('--data_dir', type=str, default=str(dataset_root('visium_libd')))
parser.add_argument('--h5ad_path', type=str, default=None)
parser.add_argument(
    '--sdeper_spatial_data_dir',
    type=str,
    default=os.environ.get(
        'LLM_ST_SDEPER_WORK_ROOT', str(run_root() / 'deconv_intermediates')
    ),
)
parser.add_argument('--run_deconv', action='store_true')
parser.add_argument('--n_threads', type=int, default=63)
args, _ = parser.parse_known_args()

print(args)

prepare_ref_data = args.prepare_ref_data
processed_dir = args.processed_dir
downsample = args.downsample

prepare_spatial_data = args.prepare_spatial_data
sample_id = args.sample_id
data_dir = os.path.join(args.data_dir, args.sample_id)
sdeper_spatial_data_dir = os.path.join(args.sdeper_spatial_data_dir, args.sample_id)

run_deconv = args.run_deconv
n_threads = args.n_threads

########################################################
# prepare reference data for deconvolution
########################################################

if prepare_ref_data:
    print("prepare reference data for deconvolution")
    output_dir = os.path.join(processed_dir, 'ref_data_for_deconv')
    # create output dir
    os.makedirs(output_dir, exist_ok=True)
    cell_type_key = 'layer_level'
    cellTypeResolution = 'layer'
    sc_path = os.path.join(processed_dir, 'adata_ref_orig.h5ad')
    marker_path = os.path.join(processed_dir, '05-shared_utilities/marker_stats_supp_table.csv')
    adata = sc.read_h5ad(sc_path)

    # # check sample and cell type distribution
    # adata.obs['Sample'].value_counts()
    # adata.obs.loc[adata.obs['Sample'] == 'Br8667_ant', 'layer_level'].value_counts()
    # adatar.obs.loc[adata.obs['layer_level'] == 'Excit_L2_3', 'Sample'].value_counts()

    print(adata.obs.loc[adata.obs['Sample'] == 'Br8667_ant', cell_type_key].value_counts())
    print("choose Br8667_ant")

    # Subset adata for 'Br8667_ant' sample
    adata_subset = adata[adata.obs['Sample'] == 'Br8667_ant'].copy()

    # Function to downsample groups
    def downsample_group(group, n=100):
        if len(group) > n:
            return group.sample(n=n, random_state=42)
        return group

    # Group by layer_level and downsample
    adata_downsampled = adata_subset.obs.groupby(cell_type_key, group_keys=False).apply(downsample_group)

    # Create a new AnnData object with downsampled observations
    if downsample:
        adata_final = adata_subset[adata_downsampled.index].copy()
    else:
        adata_final = adata_subset.copy()


    print(f"Original shape: {adata_subset.shape}")
    print(f"Downsampled shape: {adata_final.shape}")
    print("\nCell counts per layer after downsampling:")
    print(adata_final.obs[cell_type_key].value_counts())

    # marker genes: since all genes are in adata.var_names, we juse create marker matrix with adata_final
    # marker_df = pd.read_csv(marker_path)
    # marker_df = marker_df[marker_df['cellTypeResolution'] == cellTypeResolution]
    # Create a matrix with all entries as 0
    marker_matrix = pd.DataFrame(0, 
                                index=adata_final.obs[cell_type_key].unique(), 
                                columns=adata_final.var_names)


    # Save data
    marker_matrix.to_csv(os.path.join(output_dir, 'marker_matrix.csv'))
    adata_final.obs[cell_type_key].to_csv(os.path.join(output_dir, 'ref_cell_type.csv'))
    ref_expression_matrix = pd.DataFrame(adata_final.X.toarray(), 
                                        index=adata_final.obs_names, 
                                        columns=adata_final.var_names)
    # Save the expression matrix as a compressed CSV file
    ref_expression_matrix.to_csv(os.path.join(output_dir, 'ref_expression_matrix.csv.gz'), compression='gzip')



########################################################
# prepare spatial data for deconvolution
########################################################

os.makedirs(sdeper_spatial_data_dir, exist_ok=True)

if prepare_spatial_data:    
    print("prepare spatial data for deconvolution")
    # read 10X data
    if args.h5ad_path != None:
        adata_sp = sc.read_h5ad(args.h5ad_path)
        adata_sp =  adata_sp[adata_sp.obs['sample_id'] == args.sample_id].copy()
        gene_id_name = 'gene_id'
    else:
        adata_sp = sc.read_visium(data_dir)
        gene_id_name = 'gene_ids'
    adata_sp = adata_sp[adata_sp.obs['in_tissue']==1].copy()
    expression_matrix = pd.DataFrame(adata_sp.X.toarray(), index=adata_sp.obs_names, columns=adata_sp.var[gene_id_name])  # gene_ids is unique but var_names is not
    # save adata_sp
    loc_df = adata_sp.obs[['array_row', 'array_col']]
    loc_df.columns = ['x', 'y']
    loc_df.to_csv(os.path.join(sdeper_spatial_data_dir, 'spot_location.csv'))
    expression_matrix.to_csv(os.path.join(sdeper_spatial_data_dir, 'expression_matrix.csv'))

########################################################
# run deconvolution
########################################################

if run_deconv:
    output_dir = os.path.join(processed_dir, 'ref_data_for_deconv')
    print("run deconvolution")
    import subprocess
    cmd = f"""runDeconvolution -q {os.path.join(sdeper_spatial_data_dir, "expression_matrix.csv")} \
                          -r {os.path.join(output_dir, "ref_expression_matrix.csv.gz")} \
                          -c {os.path.join(output_dir, "ref_cell_type.csv")} \
                          --n_hv_gene 200 \
                          --n_marker_per_cmp 20 \
                          --pseudo_spot_max_cell 10 \
                          --lambda_r 0.72 \
                          --lambda_g 0 \
                          --seed 1 \
                          --diagnosis true \
                          --filter_cell false \
                          --filter_gene false \
                          --cvae_train_epoch 500 \
                          --use_fdr false \
                          -n {n_threads}"""

    subprocess.run(cmd, check=True, text=True, shell=True)

    # rename the output file: celltype_proportions.csv to celltype_proportions_{sample_id}.csv
    os.rename("celltype_proportions.csv", f"celltype_proportions_{sample_id}.csv")
    # rename the output dir: diagnosis to diagnosis_{sample_id}
    os.rename("diagnosis", f"diagnosis_{sample_id}")
