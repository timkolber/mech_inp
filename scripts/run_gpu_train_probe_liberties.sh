#!/bin/bash
#SBATCH --job-name=mechinp_train_libprobe
#SBATCH --output=mechinp_train_libprobe_%j.out
#SBATCH --error=mechinp_train_libprobe_%j.err
#SBATCH --partition=students
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G


srun python ./src/train_probe.py --probe_category "liberty" --num_train_games 50000 --num_classes 4