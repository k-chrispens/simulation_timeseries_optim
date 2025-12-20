"""Batched training for large datasets that don't fit in GPU memory."""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
from loguru import logger

from simulation_timeseries_optim.metrics import compute_diffuse_intensity, pearson_cc
from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.train import soft_threshold


def train_batched_hkl(
    model: Weights,
    F_array_cpu: list[np.ndarray],
    y_cpu: np.ndarray,
    n_steps: int,
    learning_rate: float,
    hkl_batch_size: int = 10000,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    use_proximal: bool = False,
    eval_every: int = 10,
    seed: int = 0,
) -> tuple[Weights, Float[Array, " n_steps"]]:
    """
    Training with mini-batching over reflections (HKL).

    Keeps data on CPU and transfers batches to GPU for each step.
    Useful when full dataset doesn't fit in GPU memory.

    Args:
        model: Initial Weights model.
        F_array_cpu: List of numpy arrays (one per dataset), kept in CPU memory.
        y_cpu: Ground truth intensities as numpy array.
        n_steps: Number of optimization steps.
        learning_rate: Learning rate.
        hkl_batch_size: Number of reflections per batch.
        lambda_l1: L1 regularization strength.
        lambda_l2: L2 regularization strength.
        use_proximal: Use proximal gradient descent.
        eval_every: Evaluate on held-out set every N steps.
        seed: Random seed for batch sampling.

    Returns:
        Tuple of (final_model, losses).
    """
    n_reflections = y_cpu.shape[0]
    key = jax.random.PRNGKey(seed)

    logger.info(
        f"Batched HKL training: {n_reflections} reflections, "
        f"batch size {hkl_batch_size}"
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(cast(optax.Params, model))

    # Create evaluation subset (fixed for consistent monitoring)
    eval_size = min(50000, n_reflections)
    F_eval = jnp.stack([jnp.array(d[:eval_size]) for d in F_array_cpu])
    y_eval = jnp.array(y_cpu[:eval_size])

    losses = []

    for step in range(n_steps):
        # Sample random batch of reflections
        key, subkey = jax.random.split(key)
        batch_indices = jax.random.choice(
            subkey, n_reflections, shape=(hkl_batch_size,), replace=False
        )
        batch_indices_np = np.array(batch_indices)

        # Transfer batch to GPU
        F_batch = jnp.stack([jnp.array(d[batch_indices_np]) for d in F_array_cpu])
        y_batch = jnp.array(y_cpu[batch_indices_np])

        # Compute loss and gradients
        def batch_loss_fn(model):
            """Compute loss for the current batch."""
            weights = model()
            I_diff = compute_diffuse_intensity(weights, F_batch)
            cc = pearson_cc(I_diff, y_batch)
            loss = -cc
            if lambda_l2 > 0:
                loss += lambda_l2 * jnp.mean((weights - 1.0) ** 2)
            if not use_proximal and lambda_l1 > 0:
                loss += lambda_l1 * jnp.mean(jnp.abs(weights - 1.0))
            return loss

        loss_value, grads = jax.value_and_grad(batch_loss_fn)(model)

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)

        if use_proximal:
            new_params = soft_threshold(model.params, learning_rate * lambda_l1)
            new_params = jnp.maximum(new_params, 0.0)
            # Update model params
            model = eqx.tree_at(lambda m: m.params, model, new_params)

        # Evaluate on held-out set
        if step % eval_every == 0:
            weights = model()
            I_diff = compute_diffuse_intensity(weights, F_eval)
            eval_loss = -pearson_cc(I_diff, y_eval)
            losses.append(float(eval_loss))
            if step % (eval_every * 10) == 0:
                logger.info(f"Step {step}/{n_steps}: eval_loss = {eval_loss:.6f}")
        else:
            losses.append(float(loss_value))

    return model, jnp.array(losses)
