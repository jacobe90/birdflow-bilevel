from flow_model import FlowModel, model_forward

import haiku as hk
import optax

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, value_and_grad, grad, vmap
from scipy.spatial.distance import pdist, squareform

from ott.solvers import linear
from ott.geometry import pointcloud, geometry
from ott import utils
from jaxtyping import Array, Float, Int
from ott.solvers.linear.implicit_differentiation import ImplicitDiff

Datatuple = namedtuple('Datatuple', ['weeks', 'ncol', 'nrow', 'cells', 'distances', 'masks', 'big_mask'])


def process_data(data_array):
    weeks, y_dim, x_dim = data_array.shape
    
    flat_data_array = data_array.reshape(weeks, -1)
    nans = jnp.isnan(flat_data_array[0])

    mass = flat_data_array[:, ~nans]
    reg = mass.sum(axis=1)
    density = mass / reg[:, None]

    cells = density.shape[1]
    dtuple = Datatuple(weeks, x_dim, y_dim, cells, nans)
    return density, dtuple


def mask_input(true_densities, dtuple):
    
    distance_matrix = jnp.zeros((dtuple.cells, dtuple.cells))
    distance_matrix = distance_matrix.at[jnp.triu_indices(dtuple.cells, k=1)].set(dtuple.distances)
    distance_matrix = distance_matrix + distance_matrix.T

    distance_matrices = []
    for i in range(0, dtuple.weeks - 1):
        distance_matrices.append(distance_matrix[dtuple.masks[i], :][:, dtuple.masks[i + 1]])

    distance_matrices_for_week = []
    for i in range(dtuple.weeks):
        distance_matrices_for_week.append(distance_matrix[dtuple.masks[i], :][:, dtuple.masks[i]])

    masked_densities = []
    for density, mask in zip(true_densities, dtuple.masks):
        masked_densities.append(density[mask])
    
    # initialize coordinate grid
    y = jnp.arange(0, dtuple.nrow, 1)
    x = jnp.arange(0, dtuple.ncol, 1)
    xv, yv = jnp.meshgrid(x, y)
    xy = jnp.stack([xv.flatten(), yv.flatten()], axis=1)
    coord_grid = xy[dtuple.big_mask, :]
    
    # get list of coordinates of cells for each week
    coordinates_for_week = []
    for mask in dtuple.masks:
        coordinates_for_week.append(coord_grid[mask])
    
    return distance_matrices, distance_matrices_for_week, masked_densities

sinkhorn_solver = jit(linear.solve, static_argnames=['max_iterations', 'progress_fn'])

# class CustomCostMatrix(CostFn):
#     def __init__(self):
#         pass
#     def __call__(self, x, y):
#         pass
#     def __call__()

def w2_obs_loss(pred_densities, true_densities, d_matrices_for_week):
    w2_obs = 0
    count = 0
    for pred, true, d_matrix in zip(pred_densities, true_densities, d_matrices_for_week):
        geom = geometry.Geometry(cost_matrix=d_matrix, epsilon=None)  # TODO: get matrix of distances between points for a given week 
        ot = sinkhorn_solver(geom, implicit_diff=ImplicitDiff(), a=pred, b=true, max_iterations=5000)
        w2_obs += ot.reg_ot_cost
        count += 1
        print(f"computed w2 loss for week {count}")
    return w2_obs

def obs_loss(pred_densities, true_densities):
    obs = 0
    for pred, true in zip(pred_densities, true_densities):
        residual = true - pred
        obs += jnp.sum(jnp.square(residual))
    return obs

def distance_loss(flows, d_matrices):
    dist = 0
    for flow, d_matrix in zip(flows, d_matrices):
        dist += jnp.sum(flow * d_matrix)
    return dist

def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h

def ent_loss(probs, flows):
    ent = 0
    for p in probs:
        ent += entropy(p)
    for f in flows:
        ent -= entropy(f)
    return ent

def w2_loss_fn(params, cells, true_densities, d_matrices, d_matrices_for_week, obs_weight, dist_weight, ent_weight):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]
    
    obs = w2_obs_loss(pred_densities, true_densities, d_matrices_for_week)
    dist = distance_loss(flows, d_matrices)
    ent = ent_loss(flows, pred_densities)
    
    return (obs_weight * obs) + (dist_weight * dist) + (-1 * ent_weight * ent), (obs, dist, ent)

def loss_fn(params, cells, true_densities, d_matrices, obs_weight, dist_weight, ent_weight):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]
    
    obs = obs_loss(pred_densities, true_densities)
    dist = distance_loss(flows, d_matrices)
    ent = ent_loss(flows, pred_densities)
    
    return (obs_weight * obs) + (dist_weight * dist) + (-1 * ent_weight * ent), (obs, dist, ent)

def train_model(loss_fn,
                optimizer,
                training_steps,
                cells,
                weeks,
                key):
    params = model_forward.init(next(key), cells, weeks)
    opt_state = optimizer.init(params)

    def update(params, opt_state):
        loss, grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    update = jit(update)

    loss_dict = {
        'total' : [],
        'obs' : [],
        'dist' : [],
        'ent' : [],
    }

    for step in range(training_steps):
        print(f"step {step}")
        params, opt_state, loss = update(params, opt_state)
        total_loss, loss_components = loss
        obs, dist, ent = loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))
    
    return params, loss_dict