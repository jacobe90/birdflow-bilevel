#!/bin/bash
#SBATCH -c 3
#SBATCH --mem=12GB
#SBATCH -p cpu
#SBATCH -t 6:00:00
#SBATCH -o slurm-%j_%a.out

module load r-rocker-ml-verse/4.4.0+apptainer
Rscript /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/validation/calculate_LL.R