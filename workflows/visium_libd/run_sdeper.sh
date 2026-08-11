#!/bin/bash
#SBATCH --job-name=run_sdeper_151509
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=46
#SBATCH --mem=16G
#SBATCH --partition=day
#SBATCH --output="%x-%j.out"

module load miniconda
conda activate sdeper-env
sample_id="151509"
: "${LLM_ST_VISIUM_ROOT:?Set LLM_ST_VISIUM_ROOT to the spatialLIBD directory}"
: "${LLM_ST_DECONV_REFERENCE_ROOT:?Set LLM_ST_DECONV_REFERENCE_ROOT}"
data_dir="${LLM_ST_VISIUM_ROOT}/${sample_id}"
ref_dir="${LLM_ST_DECONV_REFERENCE_ROOT}/ref_data_for_deconv"


runDeconvolution -q ${data_dir}/expression_matrix.csv \
                          -r ${ref_dir}/ref_expression_matrix.csv.gz \
                          -c ${ref_dir}/ref_cell_type.csv \
                          -m ${ref_dir}/marker_matrix.csv \
                          --n_hv_gene 200 \
                          --n_marker_per_cmp 20 \
                          --pseudo_spot_max_cell 10 \
                          --lambda_r 0.72 \
                          --lambda_g 0 \
                          --seed 1 \
                          --diagnosis true \
                          --filter_cell false \
                          --filter_gene false \
                          --cvae_train_epoch 300 \
                          -n 46
