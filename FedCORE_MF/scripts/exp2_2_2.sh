#!/bin/bash

# Define the parameters
datasets=("ML100KCF")
lrs=("0.40")
copies=("1")
gpus=("0" "1")
clipnum=("0.06")
epslist=("10")
attacktpye=("1" "2" "3" "4") 

# Function to run a single command and retry on failure
run_command() {
  local dataset="$1"
  local lr="$2"
  local copy="$3"
  local gpu_index="$4"
  local type="$5"
  local clip="$6"
  local eps="$7"
  
  # Calculate the actual gpu value based on the index
  local gpu="${gpus[gpu_index]}"

  # Run the command
  python exp2_2_step2.py  --dataset="$dataset"  --lr="$lr" --copy="$copy" --attacktpye="$type" --gpu="$gpu"  --clipnum="$clip" --eps="$eps"
  local exit_code=$?

  # Retry on failure
  while [ $exit_code -ne 0 ]; do
    echo "Command failed: python exp2_2_step2.py  --dataset=$dataset --lr=$lr --copy=$copy --attacktpye=$type --gpu=$gpu --clipnum=$clip --eps=$eps"
    echo "Retrying..."
    python exp2_2_step2.py  --dataset="$dataset"  --lr="$lr" --copy="$copy" --attacktpye="$type" --gpu="$gpu"  --clipnum="$clip" --eps="$eps"
    exit_code=$?
  done
}



pall=0
max_pall=12
gpuindex=0
# Loop through the parameters and run the commands in parallel
for dataset in "${datasets[@]}"; do
  for lr in "${lrs[@]}"; do
    for copy in "${copies[@]}"; do
      for type in "${attacktpye[@]}"; do 
        for clip in "${clipnum[@]}"; do
          for eps in "${epslist[@]}"; do

            echo run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]}" "$type" "$clip" "$eps" &
            run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]} " "$type" "$clip" "$eps" &
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
done

# Wait for all background jobs to finish
wait
