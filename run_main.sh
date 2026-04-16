#!/bin/bash
#SBATCH --qos=debug
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name="ensemble"
#SBATCH --chdir=.
#SBATCH --output=outs2/out_%j.txt
#SBATCH --error=errs2/err_%j.txt

module purge
module load miniforge
source activate deepLearning

python main1_ensemble.py