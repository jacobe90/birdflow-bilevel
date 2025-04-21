#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=10GB
#SBATCH -p gpu
#SBATCH -G 3
#SBATCH -t 6:00:00
#SBATCH -o slurm-%A_%a.out

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/w2_model_grid_search.py