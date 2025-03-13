import haiku as hk
import h5py
import os
import sys
sys.path.append(os.path.abspath("birdflow/birdflow-bilevel/src/"))
from flow_model_training import loss_fn, mask_input, Datatuple, w2_loss_fn
from flow_model import model_forward
from hdfs import get_plot_parameters
import numpy as np
import optax
from functools import partial
from jax import jit, grad, value_and_grad
import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.geometry.geometry import Geometry
from ott.geometry.costs import CostFn
import jax
import pickle as pkl

def train_model_and_compute_w2_loss(loss_fn,
                was_loss_fn,
                optimizer,
                training_steps,
                cells,
                weeks,
                key):
    params = model_forward.init(next(key), cells, weeks)
    opt_state = optimizer.init(params)

    def update(params, opt_state):
        loss, grads = value_and_grad(loss_fn, has_aux=True)(params)
        w2_loss, w2_grad = value_and_grad(was_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, w2_loss, w2_grad

    update = jit(update)

    loss_dict = {
        'total' : [],
        'obs' : [],
        'dist' : [],
        'ent' : [],
    }
    
    loss_dict_w2 = {
        'total' : [],
        'obs' : [],
        'grad': []
    }

    for step in range(training_steps):
        print(f"step {step}")
        params, opt_state, loss, was_loss, was_grad = update(params, opt_state)
        total_loss, loss_components = loss
        obs, dist, ent = loss_components
        total_w2_loss, w2_loss_components = was_loss
        w2_obs, _, _ = w2_loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))
        loss_dict_w2['total'].append(float(total_w2_loss))
        loss_dict_w2['obs'].append(float(w2_obs))
        loss_dict_w2['grad'].append(float(jnp.sum(was_grad**2)))
    
    return params, loss_dict, w2_loss_dict

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

# Instantiate w2 loss function
obs_weight = 1
dist_weight = 0.5
ent_weight = 0.5
w2_loss_fn = jit(partial(w2_loss_fn,
                      cells=cells,
                      true_densities=masked_densities,
                      d_matrices=distance_matrices,
                      d_matrices_for_week=distance_matrices_for_week,
                      obs_weight=obs_weight,
                      dist_weight=dist_weight,
                      ent_weight=ent_weight))
grad_w2_loss = grad(w2_loss_fn, has_aux=True)

# instantiate standard loss function
loss_fn = jit(partial(loss_fn,
                      cells=cells,
                      true_densities=masked_densities, 
                      d_matrices=distance_matrices, 
                      obs_weight=obs_weight, 
                      dist_weight=dist_weight,
                      ent_weight=ent_weight))

# Get the random seed and optimizer
key = hk.PRNGSequence(42)
optimizer = optax.adam(1e-4)
training_steps = 5
params, loss_dict, w2_loss_dict = train_model_and_compute_w2_loss(loss_fn, w2_loss_fn, optimizer, training_steps, cells, weeks, key)

# # save to pkl files
experiment_dir = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results'
with open(os.path.join(experiment_dir, f'amewoo_params_39km_obs{obs_weight}_ent{ent_weight}_dist{dist_weight}_pow{dist_pow}.pkl'), 'wb') as f:
    pickle.dump(params, f)
with open(os.path.join(experiment_dir, 'standard_loss_values.pkl'), 'wb') as f:
    pickle.dump(loss_dict, f)
with open(os.path.join(experiment_dir, 'w2_loss_values_and_grads.pkl'), 'wb') as f:
    pickle.dump(w2_loss_dict, f)
