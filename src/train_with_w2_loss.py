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
import datetime
from jax.nn import softmax

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

    for step in tqdm(range(training_steps), desc="Training Steps", unit="step"):
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
dist_weight = 1e-2
ent_weight = 1e-4
dist_pow = 0.4
dont_normalize = False
learning_rate = 0.1
training_steps = 10
rng_seed = 42
save_pkl = False
weeks = 26

# parameters for epsilon schedulers
start = 0.01
final = 0.01
decay_after = 1000
decay_iters = 100

hdf_src = os.path.join(root, f'{species}_{ebirdst_year}_{resolution}km.hdf5')
hdf_dst = os.path.join(out_dir, f'w2_{weeks}w_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.hdf5')

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

# instantiate w2 loss function
schedulers = get_epsilon_schedulers(distance_matrices_for_week, start, final, decay_iters, decay_after)

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
    with open(os.path.join(out_dir, f'ex46_w2_params_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, f'ex46_w2_losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(loss_dict, f)

# save to hdf5 file
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

    file.close()