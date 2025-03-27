import haiku as hk
import h5py
import os
import sys
from flow_model_training import loss_fn, mask_input, Datatuple, train_model, w2_loss_fn, get_epsilons, get_epsilon_schedulers
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
from typing import Any, Union, Tuple
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


def train_model_w2(loss_fn,
                   w2_loss_fn,
                   training_steps,
                   key,
                   cells,
                   weeks,
                   optimizer,
                   schedulers: Tuple):
    
    params = model_forward.init(next(key), cells, weeks)
    opt_state = optimizer.init(params)

    def get_eps_arr(step, schedulers):
        return jax.tree_util.tree_map(lambda sched: sched.get_epsilon(step), schedulers)

    def update(params, opt_state, step):
        epsilons = get_eps_arr(step, schedulers)
        loss, grads = value_and_grad(w2_loss_fn, has_aux=True, argnums=0)(params, epsilons=epsilons)
        l2_loss = loss_fn(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, l2_loss
    
    update = jit(update)

    loss_dict = {
        'total' : [],
        'w2_obs' : [],
        'dist' : [],
        'ent' : [],
        'l2_obs': [],
    }

    for step in range(training_steps):
        print(f"step {step}")
        params, opt_state, loss, l2_loss = update(params, opt_state, step)
        total_loss, loss_components = loss
        l2_total_loss, l2_loss_components = l2_loss
        obs, dist, ent = loss_components
        l2_obs, _, _ = l2_loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['w2_obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))
        loss_dict['l2_obs'].append(float(l2_obs))
    return params, loss_dict


root = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/ebird-data-loading/'
out_dir = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results'
species = 'amewoo'
ebirdst_year = 2021
resolution = 100 
obs_weight = 1.0
dist_weight = 0
ent_weight = 0
dist_pow = 0.4
dont_normalize = False
learning_rate = 0.1
training_steps = 600
rng_seed = 42
save_pkl = True
weeks = 26

# parameters for epsilon schedulers
start = 2
final = 0.01
decay_after = 500
decay_iters = 100

hdf_src = os.path.join(root, f'{species}_{ebirdst_year}_{resolution}km.hdf5')
hdf_dst = os.path.join(out_dir, f'{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.hdf5')

shutil.copyfile(hdf_src, hdf_dst)

file = h5py.File(hdf_dst, 'r+')

true_densities = np.asarray(file['distr']).T[:weeks]
total_cells = true_densities.shape[1]

distance_vector = np.asarray(file['distances'])**dist_pow
if not dont_normalize:
    distance_vector *= 1 / (100**dist_pow)

ncol, nrow, dynamic_masks, big_mask = get_plot_parameters(hdf_src)
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

# instantiate w2 loss function
schedulers = get_epsilon_schedulers(distance_matrices_for_week, start, final, decay_after, decay_iters)

w2_loss_fn = jit(partial(w2_loss_fn,
                         cells=cells,
                         true_densities=masked_densities,
                         d_matrices=distance_matrices,
                         d_matrices_for_week=distance_matrices_for_week,
                         obs_weight=obs_weight,
                         dist_weight=dist_weight,
                         ent_weight=ent_weight))

# Run Training and get params and losses
params, loss_dict = train_model_w2(loss_fn,
                                    w2_loss_fn,
                                    training_steps,
                                    key,
                                    cells,
                                    dtuple.weeks,
                                    optimizer,
                                    schedulers)

if save_pkl:
    with open(os.path.join(out_dir, f'ex41_w2_params_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, f'ex41_w2_losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(loss_dict, f)
