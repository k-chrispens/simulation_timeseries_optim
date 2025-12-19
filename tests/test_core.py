import jax
import jax.numpy as jnp
import numpy as np

from simulation_timeseries_optim.metrics import compute_diffuse_intensity, pearson_cc
from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.train import train


def test_pearson_cc():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100,))
    y = x + 0.1 * jax.random.normal(key, (100,))

    cc_jax = pearson_cc(x, y)
    cc_np = np.corrcoef(x, y)[0, 1]

    assert np.allclose(cc_jax, cc_np, atol=1e-5)


def test_compute_diffuse_intensity():
    n_time = 5
    n_hkl = 10
    weights = jnp.ones(n_time) / n_time
    F_array = jnp.ones((n_time, n_hkl), dtype=jnp.complex64)

    # If all F are 1 and weights sum to 1:
    # <F> = 1, |<F>|^2 = 1
    # <|F|^2> = 1
    # I_diff = 1 - 1 = 0

    I_diff = compute_diffuse_intensity(weights, F_array)
    assert jnp.allclose(I_diff, 0.0, atol=1e-5)


def test_weights_model():
    model = Weights(n_timepoints=10, method="softmax")
    w = model()
    assert jnp.allclose(jnp.sum(w), 1.0)
    assert w.shape == (10,)

    model = Weights(n_timepoints=10, method="sigmoid")
    w = model()
    assert jnp.all((w >= 0) & (w <= 1))


def test_training_loop():
    n_time = 5
    n_hkl = 100
    key = jax.random.PRNGKey(0)

    F_array = jax.random.normal(key, (n_time, n_hkl)) + 1j * jax.random.normal(
        key, (n_time, n_hkl)
    )
    y = jnp.abs(jax.random.normal(key, (n_hkl,)))  # Dummy intensity

    model = Weights(n_timepoints=n_time, method="raw")

    final_model, losses = train(
        model,
        F_array,
        y,
        n_steps=10,
        learning_rate=0.01,
        use_proximal=True,
        lambda_l1=0.1,
    )

    assert len(losses) == 10
    assert not jnp.isnan(losses).any()
