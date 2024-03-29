#!/bin/bash

# Define the parameters
datasets=("ML100KCF")
testtype=("test")
ratios=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
lrs=("0.40")
copies=("1" "2" "3" "4" "5")
gpus=("0" "1")

# Function to run a single command and retry on failure
run_command() {
  local dataset="$1"
  local ratio="$2"
  local lr="$3"
  local copy="$4"
  local gpu_index="$5"
  local type="$6"
  
  # Calculate the actual gpu value based on the index
  local gpu="${gpus[gpu_index]}"
  
  # Run the command
  python run_co.py --dataset="$dataset" --ratio="$ratio" --lr="$lr" --copy="$copy" --gpu="$gpu" --type="$type"
  local exit_code=$?

  # Retry on failure
  while [ $exit_code -ne 0 ]; do
    echo "Command failed: python run_co.py --dataset=$dataset --ratio=$ratio --lr=$lr --copy=$copy --gpu=$gpu --type=$type"
    echo "Retrying..."
    python run_co.py --dataset="$dataset" --ratio="$ratio" --lr="$lr" --copy="$copy" --gpu="$gpu" --type="$type"
    exit_code=$?
  done
}



pall=0
max_pall=12
gpuindex=0
# Loop through the parameters and run the commands in parallel
for dataset in "${datasets[@]}"; do
  for ratio in "${ratios[@]}"; do
    for lr in "${lrs[@]}"; do
      for copy in "${copies[@]}"; do
        for testtype in "${testtype[@]}"; do
          echo run_command "$dataset" "$ratio" "$lr" "$copy" "${gpus[gpuindex]}" "$testtype" &
          run_command "$dataset" "$ratio" "$lr" "$copy" "${gpus[gpuindex]}" "$testtype"&
          if (( ((pall+1) % max_pall) == 0 )); then
              wait 
          fi 
          gpuindex=$((gpuindex+1))
          gpuindex=$((gpuindex%2))
          pall=$((pall+1)) 
        done
      done
    done
  done
done

# Wait for all background jobs to finish
wait
