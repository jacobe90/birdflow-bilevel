#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=5GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/train_markov_and_measure_w2.py
