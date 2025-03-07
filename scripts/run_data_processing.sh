#!/bin/bash

#SBATCH --job-name=data_processing
#SBATCH --output=output/slurm/slurm_%j.out
#SBATCH --error=output/slurm/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.kolber@stud.uni-heidelberg.de

srun python src/main.py