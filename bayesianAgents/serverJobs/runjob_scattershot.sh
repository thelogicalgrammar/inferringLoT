#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=50:00:00

module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
conda activate LoT_recovery

python -u ../simulation.py --datasize $1 --n_participants $2 --temp $3 --type_experiment "scattershot"