"""Tests for train module."""
import jax
import jax.numpy as jnp
import numpy as np

from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.train import loss_fn, soft_threshold, train


class TestSoftThreshold:
    """Tests for soft_threshold function."""

    def test_above_threshold(self):
        """Values above threshold are reduced."""
        w = jnp.array([2.0, 3.0, 4.0])
        result = soft_threshold(w, 1.0)
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_below_threshold_becomes_zero(self):
        """Values below threshold become zero."""
        w = jnp.array([0.5, 0.3, 0.1])
        result = soft_threshold(w, 1.0)
        expected = jnp.array([0.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_negative_values(self):
        """Negative values are handled correctly."""
        w = jnp.array([-2.0, -0.5, 0.5, 2.0])
        result = soft_threshold(w, 1.0)
        expected = jnp.array([-1.0, 0.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_zero_threshold(self):
        """Zero threshold returns original values."""
        w = jnp.array([1.0, 2.0, 3.0])
        result = soft_threshold(w, 0.0)
        assert jnp.allclose(result, w)


class TestLossFn:
    """Tests for loss_fn function."""

    def test_loss_fn_basic(self):
        """Test basic loss function computation."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 100

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        loss = loss_fn(model, F_array, y, 0.0, 0.0, False)
        assert jnp.isfinite(loss)
        assert jnp.isscalar(loss)

    def test_loss_fn_with_l1(self):
        """Test loss function with L1 regularization."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        loss_no_reg = loss_fn(model, F_array, y, 0.0, 0.0, False)
        loss_with_l1 = loss_fn(model, F_array, y, 0.1, 0.0, False)

        # Loss with L1 should be higher
        assert loss_with_l1 > loss_no_reg

    def test_loss_fn_with_l2(self):
        """Test loss function with L2 regularization."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        loss_no_reg = loss_fn(model, F_array, y, 0.0, 0.0, False)
        loss_with_l2 = loss_fn(model, F_array, y, 0.0, 0.1, False)

        # Loss with L2 should be higher
        assert loss_with_l2 > loss_no_reg


class TestTrain:
    """Tests for train function."""

    def test_basic_training(self):
        """Test that training runs and reduces loss."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 100

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        final_model, losses = train(
            model=model,
            F_array=F_array,
            y=y,
            n_steps=10,
            learning_rate=0.01,
        )

        assert len(losses) == 10
        assert losses[-1] <= losses[0]  # Loss should decrease or stay same
        assert not np.any(np.isnan(losses))

    def test_with_l1_regularization(self):
        """Test training with L1 regularization."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        final_model, losses = train(
            model=model,
            F_array=F_array,
            y=y,
            n_steps=10,
            learning_rate=0.01,
            lambda_l1=0.1,
            use_proximal=True,
        )

        assert not np.any(np.isnan(losses))

    def test_with_remat(self):
        """Test training with gradient checkpointing."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        model = Weights(n_timepoints=n_mtz, method="raw")

        final_model, losses = train(
            model=model,
            F_array=F_array,
            y=y,
            n_steps=10,
            learning_rate=0.01,
            use_remat=True,
        )

        assert len(losses) == 10
        assert not np.any(np.isnan(losses))

    def test_different_methods(self):
        """Test training with different parameterization methods."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        for method in ["raw", "softmax", "sigmoid"]:
            model = Weights(n_timepoints=n_mtz, method=method)

            final_model, losses = train(
                model=model,
                F_array=F_array,
                y=y,
                n_steps=5,
                learning_rate=0.01,
            )

            assert len(losses) == 5
            assert not np.any(np.isnan(losses))

    def test_warm_start(self):
        """Test training with warm starting."""
        key = jax.random.PRNGKey(42)
        n_mtz, n_hkl = 5, 50

        F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
            key, (n_mtz, n_hkl)
        )
        y = jnp.abs(jax.random.normal(key, (n_hkl,)))

        # Train first model
        model1 = Weights(n_timepoints=n_mtz, method="raw")
        final_model1, _ = train(
            model=model1, F_array=F_array, y=y, n_steps=10, learning_rate=0.01
        )

        # Warm start second model with final params from first
        model2 = Weights(
            n_timepoints=n_mtz, method="raw", initial_params=final_model1.params
        )

        # Check that initial params match
        assert jnp.allclose(model2.params, final_model1.params)
