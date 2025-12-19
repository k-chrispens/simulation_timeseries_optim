from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Complex, Float

from simulation_timeseries_optim.metrics import compute_diffuse_intensity, pearson_cc
from simulation_timeseries_optim.models import Weights


def soft_threshold(w: Array, threshold: float) -> Array:
    """
    Apply soft thresholding (proximal operator for L1 norm).
    """
    return jnp.sign(w) * jnp.maximum(jnp.abs(w) - threshold, 0.0)


def loss_fn(
    model: Weights,
    F_array: Complex[Array, "time hkl"],
    y: Float[Array, " hkl"],
    lambda_l1: float,
    lambda_l2: float,
    use_proximal: bool,
) -> ArrayLike:
    """
    Calculates the loss.
    If use_proximal is True, L1 is omitted from the gradient calculation here.
    """
    weights = model()

    # Calculate intensity
    I_diff = compute_diffuse_intensity(weights, F_array)

    # Pearson Correlation (maximize -> minimize negative)
    cc = pearson_cc(I_diff, y)
    loss = -cc

    # L2 Regularization
    if lambda_l2 > 0:
        loss += lambda_l2 * jnp.mean((weights - 1.0) ** 2)

    # L1 Regularization (only if NOT using proximal)
    if not use_proximal and lambda_l1 > 0:
        loss += lambda_l1 * jnp.mean(jnp.abs(weights - 1.0))

    return loss


def train(
    model: Weights,
    F_array: Complex[Array, "time hkl"],
    y: Float[Array, " hkl"],
    n_steps: int,
    learning_rate: float,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    use_proximal: bool = False,
) -> tuple[Weights, Float[Array, " n_steps"]]:
    """
    Main training loop.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(cast(optax.Params, model))

    # Define the step function with proximal logic inside if possible,
    # or handle proximal update explicitly.

    @eqx.filter_jit
    def step_fn(carrier, _):
        model, opt_state = carrier

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            model, F_array, y, lambda_l1, lambda_l2, use_proximal
        )

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        if use_proximal:
            # Apply soft thresholding to the parameters
            # We assume 'params' is the leaf we want to touch.
            new_params = soft_threshold(model.params, learning_rate * lambda_l1)
            # Enforce non-negativity as per original code
            new_params = jnp.maximum(new_params, 0.0)
            model = eqx.tree_at(lambda m: m.params, model, new_params)

        return (model, opt_state), loss_value

    # Run the loop
    (final_model, final_state), losses = jax.lax.scan(
        step_fn, (model, opt_state), None, length=n_steps
    )

    return final_model, losses
