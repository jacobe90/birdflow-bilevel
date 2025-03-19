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


def pytree_norm(pytree, ord=2):
    """Computes the norm of a PyTree (default: L2 norm)."""
    leaves = jax.tree_util.tree_leaves(pytree)  # Extract all array leaves
    return jnp.sqrt(sum(jnp.linalg.norm(leaf, ord=ord)**2 for leaf in leaves))

def train_model_and_measure_w2_loss_and_grads(loss_fn,
                                              w2_loss_fn,
                                              training_steps,
                                              key,
                                              cells,
                                              weeks,
                                              optimizer,):
    params = model_forward.init(next(key), cells, weeks)
    opt_state = optimizer.init(params)

    def update(params, opt_state):
        loss, grads = value_and_grad(loss_fn, has_aux=True)(params)
        w2_loss, w2_grads = value_and_grad(w2_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, w2_loss, pytree_norm(w2_grads, ord=2)

    update = jit(update)

    loss_dict = {
        'total' : [],
        'obs' : [],
        'dist' : [],
        'ent' : [],
    }

    w2_loss_dict = {
        'total' : [],
        'obs' : [],
        'dist' : [],
        'ent' : [],
        'norm_w2_grad': []
    }

    for step in range(training_steps):
        print(f"step {step}")
        params, opt_state, loss, w2_loss, norm_w2_grad = update(params, opt_state)
        total_loss, loss_components = loss
        obs, dist, ent = loss_components
        w2_total, w2_loss_components = w2_loss
        w2_obs, w2_dist, w2_ent = w2_loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))
        w2_loss_dict['total'].append(float(w2_total))
        w2_loss_dict['obs'].append(float(w2_obs))
        w2_loss_dict['dist'].append(float(w2_dist))
        w2_loss_dict['ent'].append(float(w2_ent))
        w2_loss_dict['norm_w2_grad'].append(float(norm_w2_grad))     
    return params, loss_dict, w2_loss_dict


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
weeks = 5

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
epsilons = get_epsilons(distance_matrices_for_week) # get reg ot's reg strengths
w2_loss_fn = jit(partial(w2_loss_fn,
                         cells=cells,
                         true_densities=masked_densities,
                         d_matrices=distance_matrices,
                         d_matrices_for_week=distance_matrices_for_week,
                         epsilons=epsilons,
                         obs_weight=obs_weight,
                         dist_weight=dist_weight,
                         ent_weight=ent_weight))

# Run Training and get params and losses
params, loss_dict, w2_loss_val_and_grad_dict = train_model_and_measure_w2_loss_and_grads(loss_fn,
                                                                                         w2_loss_fn,
                                                                                         training_steps,
                                                                                         key,
                                                                                         cells,
                                                                                         dtuple.weeks,
                                                                                         optimizer,)

if save_pkl:
    with open(os.path.join(out_dir, f'params_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, f'losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(loss_dict, f)
    with open(os.path.join(out_dir, f'w2_losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(w2_loss_val_and_grad_dict, f)
