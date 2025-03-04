import haiku as hk
import h5py
import os
import sys
sys.path.append(os.path.abspath("birdflow/birdflow-bilevel/src/"))
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
import jax

hdf_src = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/ebird-data-loading/amewoo_2021_39km.hdf5'
file = h5py.File(hdf_src, 'r+')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
total_cells = true_densities.shape[1]

dist_pow = 0.1
distance_vector = np.asarray(file['distances'])**dist_pow
distance_vector *= 1 / (100**dist_pow) # normalize the distance vector
ncol, nrow, dynamic_masks, big_mask = get_plot_parameters(hdf_src)

dtuple = Datatuple(weeks, ncol, nrow, total_cells, distance_vector, dynamic_masks, big_mask)
print(jnp.sum(jnp.asarray(true_densities[5, :])))
distance_matrices, distance_matrices_for_week, masked_densities = mask_input(true_densities, dtuple)
cells = [d.shape[0] for d in masked_densities]
print(cells)

# Instantiate loss function
obs_weight = 1
dist_weight = 0.5
ent_weight = 0.5
w2_loss_fn = partial(w2_loss_fn,
                      cells=cells,
                      true_densities=masked_densities,
                      d_matrices=distance_matrices,
                      d_matrices_for_week=distance_matrices_for_week,
                      obs_weight=obs_weight,
                      dist_weight=dist_weight,
                      ent_weight=ent_weight)

key = hk.PRNGSequence(42)
params = model_forward.init(next(key), cells, weeks)
pred = model_forward.apply(params, None, cells, weeks)
w2_loss_val = w2_loss_fn(params)
print(f"finished: w2_loss_val = {w2_loss_val}")