#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=10GB
#SBATCH -p gypsum-1080ti
#SBATCH -G 3
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/train_with_w2_loss.py
