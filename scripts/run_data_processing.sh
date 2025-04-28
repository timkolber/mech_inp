#!/bin/bash

#SBATCH --job-name=mechinp_data_processing
#SBATCH --output=mechinp_data_processing_%j.out
#SBATCH --error=mechinp_data_processing_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.kolber@stud.uni-heidelberg.de

srun python src/parse_data.py