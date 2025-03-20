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

import jax

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

class Scheduler:
    def __init__(self, eps_default, start, final, decay_iters, decay_after):
        self.eps_init = eps_default * start
        self.eps_final = eps_default * final
        self.decay = max(0, (self.eps_init - self.eps_final)) / decay_iters 
        self.decay_after = decay_after
    
    def get_epsilon(self, it: Int[Array, ""]):
        """
        At training step it, get a value of epsilon
        """
        epsilon = self.eps_init
        epsilon = jax.lax.cond(it > self.decay_after,
                     lambda t : jax.lax.cond(t - self.decay * (it - self.decay_after) > self.eps_final, 
                                             lambda s : t - self.decay * (it - self.decay_after), 
                                             lambda s: self.eps_final, 
                                             None),
                     lambda t: t,
                     epsilon)  # linear decay   
        return epsilon

def get_epsilon_schedulers(d_matrices_for_week, start, final, decay_iters, decay_after):
    eps_defaults = get_epsilons(d_matrices_for_week)
    return list(map(lambda eps: Scheduler(eps, start, final, decay_iters, decay_after), 
                    eps_defaults))

def get_epsilons(d_matrices_for_week):
    epsilons = []
    for d_matrix in d_matrices_for_week:
        geom = geometry.Geometry(cost_matrix=d_matrix, epsilon=None)
        epsilons.append(geom.epsilon)   # get the default epsilon, based on mean of cost matrix
    return epsilons

sinkhorn_solver = jit(linear.solve, static_argnames=['max_iterations', 'progress_fn'])

def w2_obs_loss(pred_densities, true_densities, d_matrices_for_week, epsilons):
    w2_obs = 0
    for pred, true, d_matrix, eps in zip(pred_densities, true_densities, d_matrices_for_week, epsilons):
        geom = geometry.Geometry(cost_matrix=d_matrix, epsilon=eps)
        ot = sinkhorn_solver(geom, implicit_diff=ImplicitDiff(), a=pred, b=true, max_iterations=5000)
        w2_obs += ot.reg_ot_cost
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

def w2_loss_fn(params, cells, true_densities, d_matrices, d_matrices_for_week, epsilons, obs_weight, dist_weight, ent_weight):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]

    obs = w2_obs_loss(pred_densities, true_densities, d_matrices_for_week, epsilons)
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
