#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=5GB
#SBATCH -p cpu
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j_%a.out

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/launch-grid-search.py