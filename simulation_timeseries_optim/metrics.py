import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def pearson_cc(x: Float[Array, " n"], y: Float[Array, " n"]) -> Float[Array, ""]:
    """
    Calculates the Pearson Correlation Coefficient between two vectors.

    Args:
        x: First vector of observations.
        y: Second vector of observations.

    Returns:
        The Pearson correlation coefficient.
    """
    mean_x = jnp.nanmean(x)
    mean_y = jnp.nanmean(y)

    # Centered variables
    xm = x - mean_x
    ym = y - mean_y

    # Covariance and variances
    numerator = jnp.nansum(xm * ym)
    denominator = jnp.sqrt(jnp.nansum(xm**2)) * jnp.sqrt(jnp.nansum(ym**2))

    return numerator / denominator


def compute_diffuse_intensity(
    weights: Float[Array, " time"], F_array: Complex[Array, " time hkl"]
) -> Float[Array, " hkl"]:
    """
    Computes the diffuse intensity from a weighted ensemble of structure factors.

    I_diff = <|F|^2> - |<F>|^2

    Args:
        weights: Weights for each timepoint/dataset.
        F_array: Complex structure factors for each timepoint and reflection.

    Returns:
        The calculated diffuse intensity.
    """
    # Ensure weights are properly broadcasted if necessary, though einsum handles it.
    # <|F|^2> = sum(w_t * |F_t|^2)
    weighted_F2 = jnp.einsum("t,t...->...", weights, jnp.abs(F_array) ** 2)

    # |<F>|^2 = |sum(w_t * F_t)|^2
    weighted_F = jnp.einsum("t,t...->...", weights, F_array)
    abs_weighted_F_sq = jnp.abs(weighted_F) ** 2

    return weighted_F2 - abs_weighted_F_sq
