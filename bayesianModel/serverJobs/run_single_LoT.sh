#!/bin/bash
#SBATCH -n 4
#SBATCH -p shared
#SBATCH -t 1:30:00

module load 2020
module load Python/3.8.2-GCCcore-9.3.0
# activate virtual environment
source ../../../venv/bin/activate

script -c \
"python -u ../model_fitting.py byLoT \
--sampler SMC \
--indexLoT $SLURM_ARRAY_TASK_ID \
--path_learningdata '../../data/learning_costs.pkl' \
--path_L '../../data/lengths_data.npy'" \
log.txt
