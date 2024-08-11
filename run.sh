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
idle_gpus_array=($(get_idle_gpus))

# Output the idle GPUs
echo "GPUs with 0% usage: ${idle_gpus_array[@]}"

