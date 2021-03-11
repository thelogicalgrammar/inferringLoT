#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 25:00:00

module load 2019
module load Python/3.6.6-intel-2019b

python -u ../simulation.py
