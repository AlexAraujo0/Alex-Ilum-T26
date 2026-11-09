#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=3
#SBATCH --mem=2G
#SBATCH --partition=cpu
#SBATCH --job-name=melhorias_paralelo
#SBATCH --output=resultado_mel-%j.out
#SBATCH --error=%x.%j.err

source ~/.bashrc
conda init
conda activate ilumpy

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE=/home/alex2610082/work/Tiamat

python -u OTM_KNN.py > $BASE/E_KNN/knn.log 2>&1 &
python -u OTM_RIDGE.py > $BASE/E_RIDGE/RIDGE.log 2>&1 &
python -u OTM_HGB.py > $BASE/E_HGB/HGB.log 2>&1 &
wait