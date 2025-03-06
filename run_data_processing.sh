#!/bin/bash -l

#SBATCH --job-name=data_processing
#SBATCH --output=output/slurm/data_processing.out
#SBATCH --error=output/slurm/data_processing.err


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1	
#SBATCH --time=24:00:00
#SBATCH --mem=32gb

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tim.kolber@stud.uni-heidelberg.de
#SBATCH --partition=single

python src/main.py