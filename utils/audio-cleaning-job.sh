#!/bin/bash
#SBATCH -w cn-m-2
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --job-name=audio-clean
#SBATCH -t 3-00:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

#SBATCH -o audio-clean.out				  # name of output file for this submission script
#SBATCH -e audio-clean.err				  # name of error file for this submission script

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load cuda/11.7 sox

# move to split dataset dir
cd /nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch/data/rwc_all/split

# run my job (e.g. matlab, python)
/nfs/guille/eecs_research/soundbendor/zontosj/opt/bin/audio-clean.sh