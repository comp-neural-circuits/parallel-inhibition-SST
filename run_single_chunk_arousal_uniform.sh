#!/bin/bash
#SBATCH --array=0-999
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu-parallel-large
#SBATCH --output=logs/slurm-%A.out
#SBATCH --error=logs/slurm-%A.err

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh

conda activate linear_theory
python run_single_point_arousal_uniform.py ${SLURM_ARRAY_TASK_ID} 1000
