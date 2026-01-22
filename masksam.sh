#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -c 32 # num cpus
#SBATCH -J masksam # job name
#SBATCH --gres=gpu:4 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -A strategic

conda activate medsam2
cd /data/rlsam2/MaskSAM/bash
conda init
conda activate medsam2
bash MaskSAM_AMOS.sh