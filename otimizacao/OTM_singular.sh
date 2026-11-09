#!/bin/bash
#SBATCH --job-name=ETR_3_corr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilumpy

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE=/home/alex2610082/work/Tiamat
mkdir -p "$BASE/E_ETR"
cd "$BASE"

python -u OTM_ETR.py > "$BASE/E_ETR/ETR_final_2.log" 2>&1