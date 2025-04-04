from absl import app
import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver
import optax
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing


def ridge_objective(params, l2reg, data):
  """Ridge objective function."""
  X_tr, y_tr = data
  residuals = jnp.dot(X_tr, params) - y_tr
  return 0.5 * jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.sum(params ** 2)


@implicit_diff.custom_root(jax.grad(ridge_objective))
def ridge_solver(init_params, l2reg, data):
  """Solve ridge regression by conjugate gradient."""
  X_tr, y_tr = data

  def matvec(u):
    return jnp.dot(X_tr.T, jnp.dot(X_tr, u))

  return linear_solve.solve_cg(matvec=matvec,
                               b=jnp.dot(X_tr.T, y_tr),
                               ridge=len(y_tr) * l2reg,
                               init=init_params,
                               maxiter=20)


# Perhaps confusingly, theta is a parameter of the outer objective,
# but l2reg = jnp.exp(theta) is an hyper-parameter of the inner objective.
def outer_objective(theta, init_inner, data):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping l2reg = jnp.exp(theta)
  # both to optimize in log-space and to ensure positivity.
  l2reg = jnp.exp(theta)
  w_fit = ridge_solver(init_inner, l2reg, (X_tr, y_tr))
  y_pred = jnp.dot(X_val, w_fit)
  loss_value = jnp.mean((y_pred - y_val) ** 2)
  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, w_fit


def main(argv):
  del argv

  # load data
  housing = fetch_california_housing()
  X, y = housing['data'], housing['target']
  X = preprocessing.normalize(X)
  # data = (X_tr, X_val, y_tr, y_val)
  data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

  # Initialize solver.
  solver = OptaxSolver(opt=optax.adam(1e-2), fun=outer_objective, has_aux=True)
  theta = 10.0
  init_w = jnp.zeros(X.shape[1])
  state = solver.init_state(theta, init_inner=init_w, data=data)

  # Run outer loop.
  for _ in range(200):
    theta, state = solver.update(params=theta, state=state, init_inner=init_w,
                                 data=data)
    # The auxiliary data returned by the outer loss is stored in the state.
    init_w = state.aux
    print(f"[Step {state.iter_num}] Validation loss: {state.value:.3f}, theta: {theta:.3f}.")


if __name__ == "__main__":
  app.run(main)