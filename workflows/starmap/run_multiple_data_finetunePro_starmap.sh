#!/bin/zsh

# Activate the environment described in environment/environment.yml before running.
: "${CONDA_PREFIX:?Activate the speak environment before running this script}"

# Print which Python is being used
echo "Using Python from: $(which python)"

# Array of data IDs
data_ids=("BZ9" "BZ14")
replicate_name=$1
replicate_name_unconserved="${replicate_name}_unconserved"
fine_tuned_data_name="BZ5"  # The data used for fine-tuning
skip_first_stage="true"

# Create logs directory if it doesn't exist
mkdir -p logs/finetunePro_${fine_tuned_data_name}_${replicate_name}

# Loop through each data ID and run in parallel
for data_id in $data_ids; do
    # Skip the fine-tuned data ID since it's used as the reference
    if [[ "${data_id}" != "${fine_tuned_data_name}" ]]; then
        echo "Launching job for data ID: ${data_id}"
        python -u workflows/starmap/two_in_one_finetunePro_starmap.py "${data_id}" "${replicate_name}" "${replicate_name_unconserved}" "${fine_tuned_data_name}" "${skip_first_stage}" > "logs/finetunePro_${fine_tuned_data_name}_${replicate_name}/${data_id}.out" 2>&1 &
    fi
done

echo "All jobs have been launched!"
echo "You can check individual progress in the logs/finetunePro_${fine_tuned_data_name}_${replicate_name} directory"


# Keep the environment activated for any following commands
