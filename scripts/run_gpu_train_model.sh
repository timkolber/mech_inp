#!/bin/bash
#SBATCH --job-name=mechinp_train
#SBATCH --output=mechinp_train_%j.out
#SBATCH --error=mechinp_train_%j.err
#SBATCH --partition=students
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1


srun python ./src/train_model.py