import haiku as hk
import h5py
import os
import sys
from flow_model_training import loss_fn, mask_input, Datatuple, train_model, w2_loss_fn, get_epsilons
from flow_model import model_forward
from hdfs import get_plot_parameters
import numpy as np
import optax
from functools import partial
from jax import jit
import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.geometry.geometry import Geometry
from ott.geometry.costs import CostFn
from ott.solvers import linear
from ott.solvers.linear.implicit_differentiation import ImplicitDiff
from jax import value_and_grad
import jax
from jaxtyping import Float, Array, Int
from typing import Any, Union
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from beartype import (
    beartype,
    BeartypeConf,
    BeartypeStrategy,
)
import shutil
from jax.nn import softmax
from datetime import datetime

# experiment parameters
root = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/ebird-data-loading/'
out_dir = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results/l2-grid-search'
hdf_dir = os.path.join(out_dir, 'hdfs')
params_dir = os.path.join(out_dir, 'params')
losses_dir = os.path.join(out_dir, 'losses')
species = 'amewoo'
ebirdst_year = 2021
resolution = 100
dont_normalize = False
learning_rate = 0.1
training_steps = 400
rng_seed = 42
save_pkl = True
weeks = 53

# set up grid search
# center around known best parameters for markov chain
# obs_weight, dist_weight, ent_weight, dist_pow) = (1.0, 0.05, 0.01, 0.2)
# obs_weights = jnp.linspace(0.75, 1.25, num=3)
# dist_weights = jnp.linspace(4e-2, 6e-2, num=3)
# ent_weights = jnp.linspace(0.05, 0.015, num=3)
# dist_pows = jnp.linspace(0.15, 0.25, num=3)

# hyperparameters_arr = []
# for ow in obs_weights:
#     for dw in dist_weights:
#         for ew in ent_weights:
#             for dp in dist_pows:
#                 hyperparameters_arr.append({'ow': ow, 'dw': dw, 'ew': ew, 'dp': dp})

# get grid search idx
# GRID_SEARCH_IDX = int(os.getenv("SLURM_ARRAY_TASK_ID"))

# set hyperparameter values
# obs_weight, dist_weight, ent_weight, dist_pow = hyperparameters_arr[GRID_SEARCH_IDX].values()
obs_weight = 1.0
ent_weight = 0.015
dist_weight = 0.05
dist_pow = 0.2

hdf_src = os.path.join(root, f'{species}_{ebirdst_year}_{resolution}km.hdf5')
hdf_dst = os.path.join(hdf_dir, f'{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.hdf5')

shutil.copyfile(hdf_src, hdf_dst)

with h5py.File(hdf_dst, 'r+') as file:

    true_densities = np.asarray(file['distr']).T[:weeks]
    total_cells = true_densities.shape[1]

    distance_vector = np.asarray(file['distances'])**dist_pow
    if not dont_normalize:
        distance_vector *= 1 / (100**dist_pow)

ncol, nrow, dynamic_masks, big_mask = get_plot_parameters(hdf_dst)
dynamic_masks = dynamic_masks[:weeks]
dtuple = Datatuple(weeks, ncol, nrow, total_cells, distance_vector, dynamic_masks, big_mask)
distance_matrices, distance_matrices_for_week, masked_densities = mask_input(true_densities, dtuple)
cells = [d.shape[0] for d in masked_densities]

# Get the random seed and optimizer
key = hk.PRNGSequence(rng_seed)
optimizer = optax.adam(learning_rate)

# Instantiate loss function
loss_fn = jit(partial(loss_fn,
                      cells=cells,
                      true_densities=masked_densities, 
                      d_matrices=distance_matrices, 
                      obs_weight=obs_weight, 
                      dist_weight=dist_weight,
                      ent_weight=ent_weight))

# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                training_steps,
                                cells,
                                weeks,
                                key)

with open(os.path.join(params_dir, f'params_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
    pickle.dump(params, f)
with open(os.path.join(losses_dir, f'losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
    pickle.dump(loss_dict, f)

with h5py.File(hdf_dst, 'r+') as file:
    t_start = 1
    t_end = len(params) # zero indexing in range and extra item cancel each other out

    # Initial distribution
    d = softmax(params["Flow_Model/Initial_Params"]["z0"])

    # Calculate marginals  "flow_amounts"
    flow_amounts = []
    for week in range(t_start, t_end):
        z = params[f'Flow_Model/Week_{week}']['z']
        trans_prop = softmax(z, axis=1)  # softmax on rows
        flow = trans_prop * d.reshape(-1, 1) # convert d to a column and multiply each row in trans_prop by the corresponding scalar in d
        flow_amounts.append(flow)
        d = flow.sum(axis=0)
        
    margs = file.create_group('marginals')
    for i, f in enumerate(flow_amounts):
        margs.create_dataset(f'Week{i+1}_to_{i+2}', data=f)

    del file['distances']
        
    del file['metadata/birdflow_model_date'] 
    file.create_dataset('metadata/birdflow_model_date', data=str(datetime.today()))

    hyper = file.create_group("metadata/hyperparameters")
    hyper.create_dataset('obs_weight', data=obs_weight)
    hyper.create_dataset('ent_weight', data=ent_weight)
    hyper.create_dataset('dist_weight', data=dist_weight)
    hyper.create_dataset('dist_pow', data=dist_pow)
    hyper.create_dataset('learning_rate', data=learning_rate)
    hyper.create_dataset('training_steps', data=training_steps)
    hyper.create_dataset('rng_seed', data=rng_seed)
    hyper.create_dataset('normalized', data=not dont_normalize)

    loss_vals = file.create_group("metadata/loss_values")
    loss_vals.create_dataset('total', data=loss_dict['total'])
    loss_vals.create_dataset('obs', data=loss_dict['obs'])
    loss_vals.create_dataset('dist', data=loss_dict['dist'])
    loss_vals.create_dataset('ent', data=loss_dict['ent'])