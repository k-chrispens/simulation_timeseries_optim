"""Tests for iterative MTZ dropping functionality."""

import jax
import jax.numpy as jnp
import numpy as np

from simulation_timeseries_optim.dropping import (
    drop_mtzs,
    identify_mtz_to_drop,
    run_single_round,
)
from simulation_timeseries_optim.state import (
    DropConfig,
    OptimizationHistory,
    RoundResult,
)


def test_identify_mtz_to_drop_basic():
    """Test basic dropping with percentile threshold."""
    weights = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    config = DropConfig(
        percentile_threshold=20.0,  # Drop bottom 20%
        min_mtz_count=3,
    )

    drop_indices = identify_mtz_to_drop(weights, config, n_remaining=10)

    # Bottom 20% of 10 items = 2 items (indices 0 and 1)
    assert len(drop_indices) == 2
    assert 0 in drop_indices
    assert 1 in drop_indices


def test_identify_mtz_to_drop_respects_min_count():
    """Test that min_mtz_count is respected."""
    weights = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
    config = DropConfig(
        percentile_threshold=80.0,  # Would drop 4/5
        min_mtz_count=3,  # But must keep at least 3
    )

    drop_indices = identify_mtz_to_drop(weights, config, n_remaining=5)

    # Should only drop 2
    assert len(drop_indices) == 2


def test_identify_mtz_to_drop_with_absolute_threshold():
    """Test that absolute minimum weight threshold is applied."""
    weights = jnp.array([0.001, 0.002, 0.1, 0.2, 0.3])
    config = DropConfig(
        percentile_threshold=10.0,  # Would normally drop only 1
        min_weight_threshold=0.05,  # But this catches 2
        min_mtz_count=1,
    )

    drop_indices = identify_mtz_to_drop(weights, config, n_remaining=5)

    # Both items below 0.05 should be dropped
    assert len(drop_indices) == 2
    assert 0 in drop_indices
    assert 1 in drop_indices


def test_drop_mtzs():
    """Test that MTZ dropping correctly slices the array."""
    F_array = jnp.array(
        [
            [1 + 0j, 2 + 0j, 3 + 0j],
            [4 + 0j, 5 + 0j, 6 + 0j],
            [7 + 0j, 8 + 0j, 9 + 0j],
            [10 + 0j, 11 + 0j, 12 + 0j],
        ]
    )

    # Drop indices 1 and 3
    reduced = drop_mtzs(F_array, [1, 3])

    assert reduced.shape == (2, 3)
    assert jnp.allclose(reduced[0], jnp.array([1 + 0j, 2 + 0j, 3 + 0j]))
    assert jnp.allclose(reduced[1], jnp.array([7 + 0j, 8 + 0j, 9 + 0j]))


def test_drop_mtzs_empty_list():
    """Test that empty drop list returns original array."""
    F_array = jnp.array(
        [
            [1 + 0j, 2 + 0j],
            [3 + 0j, 4 + 0j],
        ]
    )

    reduced = drop_mtzs(F_array, [])

    assert reduced.shape == F_array.shape
    assert jnp.allclose(reduced, F_array)


def test_optimization_history_index_tracking():
    """Test that index mapping is correctly maintained across rounds."""
    history = OptimizationHistory(
        original_n_mtz=5,
        original_mtz_files=["a.mtz", "b.mtz", "c.mtz", "d.mtz", "e.mtz"],
    )

    # Initial mapping should be [0, 1, 2, 3, 4]
    assert list(history.current_to_original) == [0, 1, 2, 3, 4]

    # Round 1: drop local indices 1 and 3 (original b.mtz and d.mtz)
    result1 = RoundResult(
        round_number=1,
        n_mtz_start=5,
        n_mtz_end=3,
        dropped_indices=[1, 3],
        dropped_original_indices=[1, 3],
        final_weights=np.array([0.2, 0.1, 0.3, 0.1, 0.3]),
        final_loss=-0.8,
        losses=np.array([-0.5, -0.6, -0.7, -0.8]),
    )
    history.add_round(result1)

    # Now should have [0, 2, 4] (a.mtz, c.mtz, e.mtz)
    assert list(history.current_to_original) == [0, 2, 4]
    assert history.get_active_files() == ["a.mtz", "c.mtz", "e.mtz"]

    # Round 2: drop local index 1 (which is original c.mtz at index 2)
    result2 = RoundResult(
        round_number=2,
        n_mtz_start=3,
        n_mtz_end=2,
        dropped_indices=[1],
        dropped_original_indices=[2],
        final_weights=np.array([0.4, 0.2, 0.4]),
        final_loss=-0.85,
        losses=np.array([-0.8, -0.82, -0.85]),
    )
    history.add_round(result2)

    # Now should have [0, 4] (a.mtz, e.mtz)
    assert list(history.current_to_original) == [0, 4]
    assert history.get_active_files() == ["a.mtz", "e.mtz"]
    assert history.get_total_dropped() == 3


def test_drop_config_defaults():
    """Test DropConfig default values."""
    config = DropConfig()

    assert config.percentile_threshold == 10.0
    assert config.min_mtz_count == 5
    assert config.max_rounds == 10
    assert config.min_weight_threshold == 0.0


def test_run_single_round():
    """Test a single optimization round."""
    key = jax.random.PRNGKey(42)
    n_mtz = 10
    n_hkl = 50

    F_array = jax.random.normal(key, (n_mtz, n_hkl)) + 1j * jax.random.normal(
        key, (n_mtz, n_hkl)
    )
    y = jnp.abs(jax.random.normal(key, (n_hkl,)))

    config = DropConfig(
        percentile_threshold=20.0,
        min_mtz_count=3,
    )
    current_to_original = np.arange(n_mtz)

    result, reduced_F = run_single_round(
        F_array=F_array,
        y=y,
        round_number=1,
        n_steps=5,
        learning_rate=0.01,
        lambda_l1=0.1,
        lambda_l2=0.0,
        use_proximal=True,
        method="raw",
        config=config,
        current_to_original=current_to_original,
    )

    assert result.round_number == 1
    assert result.n_mtz_start == 10
    assert len(result.final_weights) == 10
    assert len(result.losses) == 5
    assert not np.isnan(result.final_loss)

    # Reduced array should have fewer MTZs
    assert reduced_F.shape[0] == result.n_mtz_end
