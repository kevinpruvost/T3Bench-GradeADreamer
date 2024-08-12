#!/bin/bash

# Function to get GPU IDs with 0% usage and VRAM usage under 5%
get_idle_gpus() {
    # Initialize an empty array
    local idle_gpus=()

    # Get the list of GPU indices, their usage, and memory usage
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

    # Loop through the results
    while IFS=, read -r gpu_id usage mem_used mem_total; do
        # Trim any leading/trailing whitespace
        usage=$(echo "$usage" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)

        # Calculate the VRAM usage percentage
        mem_usage_percent=$(awk "BEGIN {print ($mem_used/$mem_total)*100}")

        # Check if GPU usage is 0% and VRAM usage is under 5%
        if [ "$usage" -eq 0 ] && (( $(echo "$mem_usage_percent < 5" | bc -l) )); then
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
    print(f"{model_name},{env},{scripts}")
END
)

# Read and display available models
models=()
echo "Available models:"
model_i=0
while IFS=, read -r model_name env script; do
    echo "Model [$model_i]: $model_name"
    models+=("$model_name,$env,$script")
    model_i=$((model_i + 1))
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
# store cwd
cwd=$(pwd)
while IFS= read -r prompt || [[ -n $prompt ]]; do
    echo "/!\ Prompt: [$prompt]"

    # checks if model has already been generated
    echo "Checking if mesh [./outputs_mesh_t3/${model_name}/${prompt// /_}/mesh.obj] already exists..."
    if [ -f "./outputs_mesh_t3/${model_name}/${prompt// /_}/mesh.obj" ]; then
        echo "Mesh already generated for prompt [$prompt]. Skipping..."
        continue
    fi

    # cut scripts to run ([script1, script2, script3] contained in $script)
    # first remove []
    script=$(echo "$script" | cut -d '[' -f2- | cut -d ']' -f1)
    # then remove ' and "
    script=$(echo "$script" | sed "s/'//g")
    # then split by ',' into an array for for loop
    IFS=',' read -r -a scripts_to_run <<< "$(echo $script | xargs)"
    for script_to_run in "${scripts_to_run[@]}"; do
        # Replace placeholders with the current prompt
        script_to_run=$(echo "$script_to_run" | sed "s/\$PROMPT/\"$prompt\"/g")
        script_to_run=$(echo "$script_to_run" | sed "s/\$GPU_ID/$chosen_gpu/g")

        # Create a log file name based on the model name, GPU ID, and prompt
        log_file="logs/${model_name}/${prompt// /_}.log"
        mkdir -p "logs/${model_name}"
        echo "Running: [$script_to_run] on GPU $chosen_gpu with environment [$env]..."
        echo "[LOG] Logging output to $log_file..."
        eval "$script_to_run" >> "$cwd/$log_file" 2>&1
    done
    echo -ne "Mesh generated!\n\n"
done < prompts_to_evaluate.txt
eval conda deactivate

echo "All prompts have been processed for the selected model."