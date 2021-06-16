#!/bin/bash
#SBATCH -J "datasize_{$1}_nparts_{$2}_temp_{$3}"
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 25:00:00

module load 2020
module load Python/3.8.2-GCCcore-9.3.0
# activate virtual environment
source ../../../venv/bin/activate

python -u ../simulation.py --datasize $1 --n_participants $2 --temp $3