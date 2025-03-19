import haiku as hk
import h5py
import os
import sys
from flow_model_training import loss_fn, mask_input, Datatuple, train_model, w2_loss_fn
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


hdf_src = os.path.join(root, f'{species}_{ebirdst_year}_{resolution}km.hdf5')
hdf_dst = os.path.join(out_dir, f'{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.hdf5')

shutil.copyfile(hdf_src, hdf_dst)

file = h5py.File(hdf_dst, 'r+')

true_densities = np.asarray(file['distr']).T


weeks = true_densities.shape[0]
total_cells = true_densities.shape[1]

distance_vector = np.asarray(file['distances'])**dist_pow
if not dont_normalize:
    distance_vector *= 1 / (100**dist_pow)
masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)

ncol, nrow, dynamic_masks, big_mask = get_plot_parameters(hdf_src)
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
                                dtuple.weeks,
                                key)

if save_pkl:
    with open(os.path.join(out_dir, f'params_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, f'losses_{species}_{ebirdst_year}_{resolution}km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
        pickle.dump(loss_dict, f)
