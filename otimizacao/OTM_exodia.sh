#!/bin/bash
#SBATCH --job-name=EXODIA_otm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500M
#SBATCH --time=04:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilumpy

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE=/home/alex2610082/work/Tiamat
mkdir -p "$BASE/E_EXODIA"
cd "$BASE"

python -u OTM_EXODIA.py > "$BASE/E_EXODIA/EXODIA_otm.log" 2>&1