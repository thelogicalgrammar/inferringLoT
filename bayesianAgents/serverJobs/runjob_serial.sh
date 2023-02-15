#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=50:00:00

module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
conda activate LoT_recovery

python -u ../simulation.py --n_participants $1 --temp $2 --type_experiment "serial" --n_experiments 4