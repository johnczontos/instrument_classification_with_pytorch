#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -A soundbendor
#SBATCH -p soundbendor
#SBATCH --job-name=MIC
#SBATCH -t 3-00:00:00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --export=ALL

#SBATCH -o logs/run_model.out
#SBATCH -e logs/run_model.err

# Declare arrays of possible values
LEARNING_RATES=(5e-3 1e-3 5e-4 1e-4)
BATCH_SIZES=(32 64 128 256 512)
HIDDEN_DIMS=(64 128 256 512)
DROPOUT_RATES=(0.2 0.3 0.4 0.5)
NUM_LAYERS=(1 2 3 4 8)

# Number of random searches
NUM_SEARCHES=128

# Path to data directory
DATA_DIR="data/rwc_all/clean/split"

# load env
source env/bin/activate

for ((i=1; i<=NUM_SEARCHES; i++)); do
    # Randomly sample from the arrays
    LR=${LEARNING_RATES[$RANDOM % ${#LEARNING_RATES[@]}]}
    BS=${BATCH_SIZES[$RANDOM % ${#BATCH_SIZES[@]}]}
    HD=${HIDDEN_DIMS[$RANDOM % ${#HIDDEN_DIMS[@]}]}
    DR=${DROPOUT_RATES[$RANDOM % ${#DROPOUT_RATES[@]}]}
    NL=${NUM_LAYERS[$RANDOM % ${#NUM_LAYERS[@]}]}
    
    # Call your Python script with the data directory and sampled parameters
    python run_model.py $DATA_DIR --learning_rate $LR --batch_size $BS --hidden_dim $HD --dropout_rate $DR --num_layers $NL
done