from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Weights(eqx.Module):
    """
    A model representing the weights for the linear combination of structure factors.
    Supports different parameterizations to enforce constraints (e.g., sum to 1, 
    non-negative).
    """

    params: Float[Array, " n_timepoints"]
    method: str = eqx.field(static=True)

    def __init__(
        self,
        n_timepoints: int,
        method: str = "softmax",
        key: Array | None = None,
        initial_params: Array | None = None,
    ):
        """
        Args:
            n_timepoints: Number of weights to learn.
            method: Parameterization method ('softmax', 'sigmoid', 'raw', 'abs').
            key: JAX PRNG key for initialization.
            initial_params: Optional initial parameter values for warm starting.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        self.method = method

        # Use provided initial parameters if available (warm start)
        if initial_params is not None:
            if len(initial_params) != n_timepoints:
                raise ValueError(
                    f"initial_params length ({len(initial_params)}) must match "
                    f"n_timepoints ({n_timepoints})"
                )
            self.params = jnp.array(initial_params)
        # Initialize parameters based on method
        elif method == "softmax":
            # Logits initialized to 0 (uniform distribution)
            self.params = jnp.zeros(n_timepoints)
        elif method == "sigmoid":
            # Logits initialized to 0 (0.5 probability)
            self.params = jnp.zeros(n_timepoints)
        elif method == "raw":
            # initialized to 1/N
            self.params = jnp.ones(n_timepoints) / n_timepoints
        elif method == "abs":
            # initialized to 1/N, will take abs() later
            self.params = jnp.ones(n_timepoints) / n_timepoints
        else:
            raise ValueError(f"Unknown method: {method}")

    def __call__(self) -> Float[Array, " n_timepoints"]:
        """
        Returns the weights based on the parameterization method.

        Returns:
            Array of weights with shape (n_timepoints,).
        """
        if self.method == "softmax":
            return jax.nn.softmax(self.params)
        elif self.method == "sigmoid":
            return jax.nn.sigmoid(self.params)
        elif self.method == "abs":
            return jnp.abs(self.params)
        elif self.method == "raw":
            return self.params
        else:
            return self.params
