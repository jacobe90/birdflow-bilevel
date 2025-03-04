#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=15GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH -o slurm-%x.%j.out

/home/jepstein_umass_edu/.conda/envs/thesis/bin/python /work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/src/compute_w2_distance.py

# FIX='false'
# CUSTOM_INIT='false'
# while getopts :e:d:p:n:s:r:o:i:k:t:fc flag
# do
#     case "${flag}" in
#         e) ENTWEIGHT=${OPTARG};;
#         d) DISTWEIGHT=${OPTARG};;
#         p) DISTPOW=${OPTARG};;
#         n) NCOMPONENTS=${OPTARG};;
#         s) SPECIES=${OPTARG};;
#         r) RES=${OPTARG};;
#         o) ROOT=${OPTARG};;
#         i) SAVEDIR=${OPTARG};;
#         k) KEYSEED=${OPTARG};;
#         t) INITIAL_PARAMS_PATH=${OPTARG};;
#         f) FIX='true';;
#         c) CUSTOM_INIT='true';;
#         *) echo "invalid command: no parameter included with argument $OPTARG";;
#     esac
# done

# if ${FIX}; then
#     /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SAVEDIR $SPECIES $RES --ent_weight $ENTWEIGHT --dist_weight $DISTWEIGHT --dist_pow $DISTPOW --num_components $NCOMPONENTS --rng_seed $KEYSEED --fix_weights
# elif ${CUSTOM_INIT}; then
#     /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SAVEDIR $SPECIES $RES --ent_weight $ENTWEIGHT --dist_weight $DISTWEIGHT --dist_pow $DISTPOW --num_components $NCOMPONENTS --rng_seed $KEYSEED --initialize_from_params --initial_params_path $INITIAL_PARAMS_PATH
# else
#     /home/jepstein_umass_edu/.conda/envs/birdflow_two/bin/python train_mixture_of_products.py $ROOT $SAVEDIR $SPECIES $RES --ent_weight $ENTWEIGHT --dist_weight $DISTWEIGHT --dist_pow $DISTPOW --num_components $NCOMPONENTS --rng_seed $KEYSEED
# fi