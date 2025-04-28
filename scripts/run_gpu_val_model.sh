#!/bin/bash
#SBATCH --job-name=mechinp_val
#SBATCH --output=mechinp_val_%j.out
#SBATCH --error=mechinp_val_%j.err

#SBATCH --partition=students
#SBATCH --gres=gpu:1


srun python ./src/validate_model.py