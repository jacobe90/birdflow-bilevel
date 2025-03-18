import haiku as hk
import h5py
import os
import sys
sys.path.append(os.path.abspath("birdflow-bilevel/src/"))
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


hdf_src = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/ebird-data-loading/amewoo_2021_39km.hdf5'

with h5py.File(hdf_src, 'r') as file:
    true_densities = np.asarray(file['distr']).T

    weeks = true_densities.shape[0]
    total_cells = true_densities.shape[1]

    dist_pow = 0.1
    distance_vector = np.asarray(file['distances'])**dist_pow
    distance_vector *= 1 / (100**dist_pow) # normalize the distance vector
    ncol, nrow, dynamic_masks, big_mask = get_plot_parameters(hdf_src)

    dtuple = Datatuple(weeks, ncol, nrow, total_cells, distance_vector, dynamic_masks, big_mask)
    distance_matrices, distance_matrices_for_week, masked_densities = mask_input(true_densities, dtuple)

eps_defaults = []
for i, distance_matrix in enumerate(distance_matrices_for_week):
    geom = Geometry(cost_matrix=distance_matrix, epsilon=None)
    eps_defaults.append(geom.epsilon)

sinkhorn_solver = jit(linear.solve, static_argnames=['max_iterations', 'progress_fn'])

class Scheduler:
    @beartype(conf=BeartypeConf(strategy=(BeartypeStrategy.O0)))
    def __init__(self, target, init, decay, decay_after):
        self.target = target
        self.init = init
        self.decay = decay
        self.decay_after = decay_after
    
    def get_epsilon(self, it: Int[Array, ""]):
        """
        At training step it, get a value of epsilon
        """
        epsilon = self.init * self.target
        epsilon = jax.lax.cond(it > self.decay_after,
                     lambda t : jax.lax.cond(t - self.decay * (it - self.decay_after) > self.target, lambda s : t - self.decay * (it - self.decay_after), lambda s: self.target, None),
                     lambda t: t,
                     epsilon)  # linear decay   
        return epsilon

def w2_distance(mu: Float[Array, "n "], mu_true: Float[Array, "n "], distance_matrix: Float[Array, "n n"], epsilon: Union[Float[Array, ""], Any]):
    geom = Geometry(cost_matrix=distance_matrix, epsilon=epsilon)
    ot = sinkhorn_solver(geom, implicit_diff=ImplicitDiff(), a=mu, b=mu_true, max_iterations=5000)
    return ot.reg_ot_cost

def loss_fn(theta: Float[Array, "n "], st_marginal: Float[Array, "n "], distance_matrix: Float[Array, "n n"], epsilon: Union[Float[Array, ""], Any]):
    mu = jax.nn.softmax(theta)
    return w2_distance(mu, st_marginal, distance_matrix, epsilon)

def logit_l2_loss(theta: Float[Array, "n "], marginal: Float[Array, "n "]):
    """
    l2 distance between logits and probability distribution
    """
    mu = jax.nn.softmax(theta)
    return jnp.sqrt(jnp.sum((mu - marginal)**2))

def learn_st_marginal(st_marginal: Float[Array, "n"], 
                      distance_matrix: Float[Array, "n n"], 
                      seed=42,
                      training_steps=500, 
                      epsilon: Union[float, Any]=None, 
                      tau: float=1.0, 
                      lr: float=1e-3, 
                      init_from_mu_true: bool=False, 
                      use_adam: bool=True, 
                      use_stabilization: bool=False, 
                      scheduler: Union[Scheduler, Any]=None,
                      loss_fn=loss_fn):
    
    solver = optax.adam(learning_rate=lr)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    # if init_from_mu_true:  # initialize at the true marginal
    #     mu_true_logits = jnp.log(mu_true)
    #     theta_init = mu_true_logits
    # else:
    #     theta_init = jnp.zeros(shape=X.shape[0]) # uniform initialization

    # initialize parameter values
    theta_init = jnp.zeros(shape=st_marginal.shape[0]) # uniform initialization    
    
    # initialize solver
    theta = theta_init
    opt_state = solver.init(theta)
    
    # create loss function
    loss_fn = functools.partial(loss_fn, st_marginal=st_marginal, distance_matrix=distance_matrix)
    grad_loss_fn = jax.value_and_grad(loss_fn)
    
    @jax.jit
    def make_step_adam(theta, epsilon, opt_state):
        loss_val, grads = grad_loss_fn(theta, epsilon=epsilon)
        updates, opt_state = solver.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        return loss_val, theta, opt_state
    
    @jax.jit
    def make_step(theta, epsilon, opt_state):
        loss_val, grads = grad_loss_fn(theta, epsilon)
        theta = theta - lr * grads
        return loss_val, theta, None
    
    @jax.jit
    def make_step_stabilized(theta, epsilon, opt_state):
        loss_val, grads = grad_loss_fn(theta, epsilon)
        theta = theta - lr * grads
        
        # stabilize by subtracting max from theta
        theta = theta - jnp.max(theta) # softmax(X - c) = softmax(X) 
        
        return loss_val, theta, None
    
    if use_adam:
        make_step = make_step_adam
    if use_stabilization:
        make_step = make_step_stabilized
    
    w2_loss_vals = []
    l2_loss_vals = []
    thetas = {}
    if scheduler != None:
        get_epsilon = jax.jit(scheduler.get_epsilon)
    for step in tqdm(range(1, training_steps + 1), desc="Training Steps", unit="step"):
        if scheduler != None:
            epsilon = float(get_epsilon(step))
        loss_val, theta, opt_state = make_step(theta, epsilon, opt_state)
        if step % int(training_steps / 40) == 0:
            thetas[step] = theta
        w2_loss_vals.append(loss_val)
        l2_loss_val = logit_l2_loss(theta, st_marginal)
        l2_loss_vals.append(l2_loss_val)
        #print(f"iteration {step}, loss value {loss_val}, l2 loss: {l2_loss_val}")
    
    return theta, thetas, (w2_loss_vals, l2_loss_vals)

week = 0
st_marginal = jnp.array(masked_densities[week])
distance_matrix = distance_matrices_for_week[week]
eps_default = eps_defaults[week]
n_steps = 2000
scheduler = Scheduler(eps_default * 0.01, 200, 9.352196e-06,2000) # decline to eps_default * 0.01 after 1000 training steps
lr = 1e-3
print("starting training")
learned, thetas, (w2_loss_vals, l2_loss_vals) = learn_st_marginal(st_marginal, distance_matrix, training_steps=n_steps, scheduler=scheduler, lr=lr)

results_obj = {'learned': learned, 'thetas': thetas, 'w2_loss_vals': w2_loss_vals, 'l2_loss_vals': l2_loss_vals}

experiment_dir = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/experiment-results'
with open(os.path.join(experiment_dir, f'learn-st-marginal-week{week}.pkl'), 'wb') as f:
    pickle.dump(results_obj, f)
