#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 25:00:00

module load 2020
module load Python/3.8.2-GCCcore-9.3.0
# Activate virtual environment
source ../../../venv/bin/activate

python -u ../simulation.py
