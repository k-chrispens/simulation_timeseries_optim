"""Tests for metrics module."""
import jax.numpy as jnp
import numpy as np

from simulation_timeseries_optim.metrics import compute_diffuse_intensity, pearson_cc


class TestPearsonCC:
    """Tests for pearson_cc function."""

    def test_perfect_correlation(self):
        """Test pearson_cc with perfectly correlated vectors."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = pearson_cc(x, y)
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_perfect_negative_correlation(self):
        """Test pearson_cc with perfectly negatively correlated vectors."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0])
        result = pearson_cc(x, y)
        assert jnp.isclose(result, -1.0, atol=1e-6)

    def test_zero_correlation(self):
        """Test pearson_cc with uncorrelated vectors."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])
        result = pearson_cc(x, y)
        assert jnp.abs(result) < 0.5

    def test_zero_variance_x(self):
        """Test pearson_cc when x has zero variance (all same value)."""
        x = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pearson_cc(x, y)
        # Should return 0 instead of NaN
        assert jnp.isfinite(result)
        assert result == 0.0

    def test_zero_variance_both(self):
        """Test pearson_cc when both have zero variance."""
        x = jnp.array([3.0, 3.0, 3.0, 3.0])
        y = jnp.array([7.0, 7.0, 7.0, 7.0])
        result = pearson_cc(x, y)
        # Should return 0 instead of NaN
        assert jnp.isfinite(result)
        assert result == 0.0

    def test_with_nans(self):
        """Test pearson_cc handles NaN values."""
        x = jnp.array([1.0, jnp.nan, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 4.0, jnp.nan, 8.0, 10.0])
        result = pearson_cc(x, y)
        assert jnp.isfinite(result)

    def test_eps_parameter(self):
        """Test pearson_cc with custom epsilon."""
        x = jnp.array([1.0, 1.0, 1.0])
        y = jnp.array([2.0, 3.0, 4.0])
        result = pearson_cc(x, y, eps=1e-10)
        assert result == 0.0


class TestComputeDiffuseIntensity:
    """Tests for compute_diffuse_intensity function."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        F_array = jnp.array([
            [1+0j, 2+0j],
            [1+1j, 2+1j],
            [1-1j, 2-1j],
            [1+0j, 2+0j],
        ])
        result = compute_diffuse_intensity(weights, F_array)
        assert result.shape == (2,)
        assert jnp.all(result >= 0)  # Diffuse intensity should be non-negative

    def test_single_weight(self):
        """Test with single non-zero weight."""
        weights = jnp.array([1.0, 0.0, 0.0])
        F_array = jnp.array([
            [1+2j, 3+4j],
            [5+6j, 7+8j],
            [9+10j, 11+12j],
        ])
        result = compute_diffuse_intensity(weights, F_array)
        # With single weight, I_diff = |F|^2 - |F|^2 = 0
        assert jnp.allclose(result, 0.0)

    def test_dtype_preservation(self):
        """Test that output is float, not complex."""
        weights = jnp.array([0.5, 0.5])
        F_array = jnp.array([[1+1j, 2+2j], [3+3j, 4+4j]])
        result = compute_diffuse_intensity(weights, F_array)
        assert jnp.issubdtype(result.dtype, jnp.floating)

    def test_shape_consistency(self):
        """Test that output shape matches HKL dimension."""
        weights = jnp.array([0.2, 0.3, 0.5])
        n_hkl = 100
        F_array = jnp.ones((3, n_hkl), dtype=jnp.complex64)
        result = compute_diffuse_intensity(weights, F_array)
        assert result.shape == (n_hkl,)

    def test_non_negative_output(self):
        """Test that diffuse intensity is always non-negative."""
        # Generate random data
        np.random.seed(42)
        weights = jnp.array([0.3, 0.7])
        F_array = jnp.array(
            np.random.randn(2, 50) + 1j * np.random.randn(2, 50),
            dtype=jnp.complex64
        )
        result = compute_diffuse_intensity(weights, F_array)
        # Due to numerical precision, may have small negative values
        assert jnp.all(result >= -1e-6)
