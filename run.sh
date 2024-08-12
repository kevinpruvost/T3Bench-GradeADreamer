#!/bin/bash

# Function to get GPU IDs with 0% usage
get_idle_gpus() {
    # Initialize an empty array
    local idle_gpus=()

    # Get the list of GPU indices and their usage, sorted by least used first
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n)

    # Loop through the results
    while IFS=, read -r gpu_id usage; do
        # Trim any leading/trailing whitespace from usage
        usage=$(echo "$usage" | xargs)

        # Check if usage is 0%
        if [ "$usage" -eq 0 ]; then
            # Add GPU ID to the idle_gpus array
            idle_gpus+=("$gpu_id")
        fi
    done <<< "$gpu_info"

    # Return the array
    echo "${idle_gpus[@]}"
}

# Call the function and store the result in an array
gpu_ids=($(get_idle_gpus))

# Get the number of idle GPUs
num_idle_gpus=${#gpu_ids[@]}

# Print the custom message
if [ "$num_idle_gpus" -gt 0 ]; then
    echo "You have $num_idle_gpus GPU(s) available/0% GPU-Utilization (IDs: ${gpu_ids[@]})"
else
    echo "No GPUs are currently available/0% GPU-Utilization."
    return 1
fi

# Parse the YAML file using Python with pyyaml
config_file="models_configs.yaml"

python_output=$(python3 - <<END
import yaml
import sys

# Load the YAML file
with open("$config_file", 'r') as file:
    config = yaml.safe_load(file)

# Print the parsed data
for model_name, model_data in config.items():
    env = model_data[0]['env']
    scripts = model_data[0]['scripts']
    for script in scripts:
        print(f"{model_name},{env},{script}")
END
)

# Read and display available models
models=()
echo "Available models:"
model_i=0
while IFS=, read -r model_name env script; do
    echo "Model [$model_i]: $model_name"
    models+=("$model_name,$env,$script")
    model_i=$model_i+1
done <<< "$python_output"

# Ask user to choose a model
read -p "Choose a model by entering the corresponding number[0-$((${#models[@]}-1))]: " model_index

# Check if the input is valid
if [ "$model_index" -ge "${#models[@]}" ]; then
    echo "Invalid model choice."
    return 1
fi
selected_model=${models[$model_index]}

if [ -z "$selected_model" ]; then
    echo "Invalid model choice."
    return 1
fi

# Parse selected model information
IFS=, read -r model_name env script <<< "$selected_model"

echo "Selected model: [$model_name]"

# Ask user to choose a GPU
echo "Available GPUs: ${gpu_ids[@]}"
read -p "Choose a GPU by entering the corresponding ID: " chosen_gpu

# Check if the GPU ID is valid
if [[ ! " ${gpu_ids[@]} " =~ " ${chosen_gpu} " ]]; then
    echo "Invalid GPU choice."
    return 1
fi

# Replace placeholders in the script
echo "$script"
script=$(echo "$script" | sed "s/\$GPU_ID/$chosen_gpu/g")

# Function to handle cleanup on exit
cleanup() {
    echo "Cleaning up..."
    pkill -P $$  # Kill all child processes of this script
}

# Run the cleanup function in the background
cleanup_handler() {
    trap 'cleanup' EXIT
    wait
}

# Start cleanup handler in background
cleanup_handler &

# Activate environment and run the script for each prompt
echo -ne "\nActivating environment '$env'...\n\n"
eval "$(conda shell.bash hook)"
eval conda activate "$env"
while IFS= read -r prompt || [[ -n $prompt ]]; do
    echo "/!\ Prompt: [$prompt]"

    # Replace placeholders with the current prompt
    script_to_run=$(echo "$script" | sed "s/\$PROMPT/\"$prompt\"/g")

    # Create a log file name based on the model name, GPU ID, and prompt
    log_file="logs/${model_name}/${prompt// /_}.log"
    mkdir -p "logs/${model_name}"
    echo "Running: $script_to_run on GPU $chosen_gpu with environment [$env]..."
    echo "[LOG] Logging output to $log_file..."
    eval "$script_to_run" > "$log_file" 2>&1
    echo -ne "Mesh generated!\n\n"
done < prompts_to_evaluate.txt
eval conda deactivate

echo "All prompts have been processed for the selected model."