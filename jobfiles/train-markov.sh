#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=10GB
#SBATCH -p gpu
#SBATCH -G 3
#SBATCH -t 6:00:00
#SBATCH -o slurm-%x.%j.out

ROOT=$1
SPECIES=$2
RES=$3

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python \
  /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/update_hdf.py \
  "$ROOT" "$SPECIES" "$RES"