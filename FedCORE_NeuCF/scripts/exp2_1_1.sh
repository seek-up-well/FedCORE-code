#!/bin/bash


# # exp2-1 table step1  ML00K
# dataset='ML100KCF'
# copylist = [1,2,3,4,5]
# gpu = [1,1,1,1,1]
# lrlist = '0.1'
# testtype = 'test'
# for i in range(len(copylist)):
#     print(f'python exp2_1table_step1.py --dataset={dataset}  --copy={copylist[i]} --type={testtype} --lr={lrlist} --gpu={gpu[i]}' + ' & ',end=" ")



# Define the parameters
datasets=("ML100KCF")
testtype=("test") 
lrs=("0.10")
copies=("1" "2" "3" "4" "5")
gpus=("2")

# Function to run a single command and retry on failure
run_command() {
  local dataset="$1"
  local lr="$2"
  local copy="$3"
  local gpu="$4"
  local type="$5"
  
  # # Calculate the actual gpu value based on the index
  # local gpu="${gpus[gpu_index]}"
  
  # Run the command
  python exp2_1table_step1.py --dataset="$dataset"  --lr="$lr" --copy="$copy" --gpu="$gpu" --type="$type"
  local exit_code=$?

  # Retry on failure
  while [ $exit_code -ne 0 ]; do
    echo "Command failed: python exp2_1table_step1.py --dataset=$dataset --lr=$lr --copy=$copy --gpu=$gpu --type=$type"
    echo "Retrying..."
    python exp2_1table_step1.py --dataset="$dataset"  --lr="$lr" --copy="$copy" --gpu="$gpu" --type="$type"
    exit_code=$?
  done
}

gpu_length=${#gpus[@]}

pall=0
max_pall=3
gpuindex=0
# Loop through the parameters and run the commands in parallel
for dataset in "${datasets[@]}"; do

  for lr in "${lrs[@]}"; do
    for copy in "${copies[@]}"; do
      for testtype in "${testtype[@]}"; do
        echo run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]}" "$testtype" &
        run_command "$dataset" "$lr" "$copy" "${gpus[gpuindex]}" "$testtype"&
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

# Wait for all background jobs to finish
wait
