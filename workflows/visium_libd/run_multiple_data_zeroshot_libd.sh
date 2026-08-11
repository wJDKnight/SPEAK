#!/bin/zsh

# Activate the environment described in environment/environment.yml before running.
: "${CONDA_PREFIX:?Activate the speak environment before running this script}"

# Print which Python is being used
echo "Using Python from: $(which python)"

# Array of data IDs
# "151507" "151508" "151509" "151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676"
data_ids=("151507" "151508" "151509" "151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676")
replicate_name=$1
# Create logs directory if it doesn't exist
mkdir -p logs/zeroshot${replicate_name}

# Loop through each data ID and run in parallel
for data_id in $data_ids; do
    echo "Launching job for data ID: ${data_id}"
    # python -u workflows/visium_libd/run_zeroshot_libd.py "${data_id}" "${replicate_name}" > "logs/zeroshot${replicate_name}/${data_id}.out" 2>&1 &
    python -u workflows/visium_libd/run_zeroshot_gemini_libd.py "${data_id}" "${replicate_name}" > "logs/zeroshot${replicate_name}/gemini_${data_id}.out" 2>&1 &
done

echo "All jobs have been launched!"
echo "You can check individual progress in the logs directory"
echo "Use 'jobs' command to see running processes"

# Keep the environment activated for any following commands
