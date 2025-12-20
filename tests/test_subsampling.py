import numpy as np

from simulation_timeseries_optim.io import subsample_reflections


def test_subsample_reflections_no_subsample():
    """Test that no subsampling occurs when fraction is 1.0."""
    n_datasets = 5
    n_reflections = 100

    datasets = [np.random.randn(n_reflections) for _ in range(n_datasets)]
    y = np.random.randn(n_reflections)

    sub_datasets, sub_y = subsample_reflections(datasets, y, subsample_fraction=1.0)

    # Should return the same arrays
    assert len(sub_datasets) == n_datasets
    assert sub_y.shape[0] == n_reflections
    np.testing.assert_array_equal(sub_y, y)


def test_subsample_reflections_half():
    """Test subsampling to 50% of reflections."""
    n_datasets = 3
    n_reflections = 100

    datasets = [np.random.randn(n_reflections) + i for i in range(n_datasets)]
    y = np.random.randn(n_reflections)

    sub_datasets, sub_y = subsample_reflections(
        datasets, y, subsample_fraction=0.5, random_seed=42
    )

    # Check shapes
    assert len(sub_datasets) == n_datasets
    assert sub_y.shape[0] == 50
    for d in sub_datasets:
        assert d.shape[0] == 50

    # Check that values are a subset (indices should be sorted)
    # We can't easily verify exact indices, but we can check the data is consistent
    assert sub_y.shape[0] < y.shape[0]


def test_subsample_reflections_deterministic():
    """Test that subsampling is deterministic with same seed."""
    n_datasets = 2
    n_reflections = 100

    datasets = [np.random.randn(n_reflections) for _ in range(n_datasets)]
    y = np.random.randn(n_reflections)

    sub1_datasets, sub1_y = subsample_reflections(
        [d.copy() for d in datasets], y.copy(), subsample_fraction=0.5, random_seed=42
    )

    sub2_datasets, sub2_y = subsample_reflections(
        [d.copy() for d in datasets], y.copy(), subsample_fraction=0.5, random_seed=42
    )

    # Should get the same subsampling
    np.testing.assert_array_equal(sub1_y, sub2_y)
    for d1, d2 in zip(sub1_datasets, sub2_datasets):
        np.testing.assert_array_equal(d1, d2)


def test_subsample_reflections_different_seed():
    """Test that different seeds produce different subsamples."""
    n_datasets = 2
    n_reflections = 100

    datasets = [np.random.randn(n_reflections) for _ in range(n_datasets)]
    y = np.random.randn(n_reflections)

    sub1_datasets, sub1_y = subsample_reflections(
        [d.copy() for d in datasets], y.copy(), subsample_fraction=0.5, random_seed=42
    )

    sub2_datasets, sub2_y = subsample_reflections(
        [d.copy() for d in datasets], y.copy(), subsample_fraction=0.5, random_seed=123
    )

    # Should get different subsampling (with very high probability)
    assert not np.allclose(sub1_y, sub2_y)


def test_subsample_reflections_indices_sorted():
    """Test that indices are kept sorted for better memory access."""
    n_reflections = 100
    datasets = [np.arange(n_reflections)]
    y = np.arange(n_reflections)

    sub_datasets, sub_y = subsample_reflections(
        datasets, y, subsample_fraction=0.5, random_seed=42
    )

    # Since indices are sorted, subsampled values should be in increasing order
    assert np.all(np.diff(sub_y) > 0), "Subsampled indices should be sorted"
