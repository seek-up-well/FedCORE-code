#!/bin/bash

# Define the parameters
datasets=("ML100KCF")
testtype=("test") 
lrs=("0.10")
copies=("1" "2" "3" "4" "5")
gpus=("2" "3")
clipnum=("0.1" "0.03" "0.07" "0.05")
epslist=("0.5" "1" "3" "5" "8" "10" "12" "15" "20" "30" "40" "50" "60" "70" "80" "10000000000")
gpu_length=${#gpus[@]}
# Function to run a single command and retry on failure
run_command() {
  local dataset="$1"
  local lr="$2"
  local copy="$3"
  local gpu="$4"
  local type="$5"
  local clip="$6"
  local eps="$7"
  
  # Calculate the actual gpu value based on the index
  # local gpu="${gpus[gpu_index]}"

  # Run the command
  python exp2_1_5_step1.py  --dataset="$dataset"  --lr="$lr" --copy="$copy" --type="$type" --gpu="$gpu"  --clipnum="$clip" --eps="$eps"
  local exit_code=$?

}



pall=0
max_pall=10

gpuindex=0
# Loop through the parameters and run the commands in parallel
for dataset in "${datasets[@]}"; do
  for lr in "${lrs[@]}"; do
    for copy in "${copies[@]}"; do
      for testtype in "${testtype[@]}"; do 
        for clip in "${clipnum[@]}"; do
          for eps in "${epslist[@]}"; do

            echo run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]}" "$testtype" "$clip" "$eps" &
            run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]} " "$testtype" "$clip" "$eps" &
            if (( ((pall+1) % max_pall) == 0 )); then
                wait 
            fi 
            gpuindex=$((gpuindex+1))
            gpuindex=$((gpuindex%gpu_length))
            pall=$((pall+1)) 
          done
        done
      done
    done
  done
done
# Wait for all background jobs to finish
wait
