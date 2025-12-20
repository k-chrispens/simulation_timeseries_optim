"""Tests for io module."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from simulation_timeseries_optim.io import load_datasets, load_structure_factors


class TestLoadStructureFactors:
    """Tests for load_structure_factors function."""

    def test_load_parquet_real_imag(self, tmp_path):
        """Test loading parquet with f_real, f_imag columns."""
        df = pd.DataFrame({
            "H": [1, 2, 3],
            "K": [0, 0, 0],
            "L": [0, 0, 0],
            "f_real": [1.0, 2.0, 3.0],
            "f_imag": [0.5, 1.0, 1.5],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0), (3, 0, 0)], names=["H", "K", "L"]
        )
        result = load_structure_factors(str(path), valid_indices, "FC,PHIC")

        assert result.shape == (3,)
        assert np.iscomplexobj(result)
        assert result.dtype == np.complex64

    def test_load_parquet_amp_phase(self, tmp_path):
        """Test loading parquet with amplitude and phase columns."""
        df = pd.DataFrame({
            "H": [1, 2],
            "K": [0, 0],
            "L": [0, 0],
            "FC": [1.0, 2.0],
            "PHIC": [0.0, np.pi/2],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0)], names=["H", "K", "L"]
        )
        result = load_structure_factors(str(path), valid_indices, "FC,PHIC")

        assert result.shape == (2,)
        # First: 1.0 * exp(0) = 1+0j
        assert np.isclose(result[0], 1.0 + 0j)

    def test_missing_columns_raises(self, tmp_path):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "H": [1],
            "K": [0],
            "L": [0],
            "wrong_col": [1.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples([(1, 0, 0)], names=["H", "K", "L"])
        with pytest.raises(ValueError, match="Unable to locate"):
            load_structure_factors(str(path), valid_indices, "FC,PHIC")

    def test_dtype_complex64(self, tmp_path):
        """Test that complex64 dtype is used correctly."""
        df = pd.DataFrame({
            "H": [1, 2],
            "K": [0, 0],
            "L": [0, 0],
            "f_real": [1.0, 2.0],
            "f_imag": [0.5, 1.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0)], names=["H", "K", "L"]
        )
        result = load_structure_factors(
            str(path), valid_indices, "FC,PHIC", dtype=np.complex64
        )

        assert result.dtype == np.complex64

    def test_dtype_complex128(self, tmp_path):
        """Test that complex128 dtype is used correctly."""
        df = pd.DataFrame({
            "H": [1, 2],
            "K": [0, 0],
            "L": [0, 0],
            "f_real": [1.0, 2.0],
            "f_imag": [0.5, 1.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0)], names=["H", "K", "L"]
        )
        result = load_structure_factors(
            str(path), valid_indices, "FC,PHIC", dtype=np.complex128
        )

        assert result.dtype == np.complex128


class TestLoadDatasets:
    """Tests for load_datasets function."""

    def test_load_multiple_files(self, tmp_path):
        """Test loading multiple parquet files."""
        files = []
        for i in range(3):
            df = pd.DataFrame({
                "H": [1, 2],
                "K": [0, 0],
                "L": [0, 0],
                "f_real": [float(i+1), float(i+2)],
                "f_imag": [0.5, 1.0],
            })
            path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(path)
            files.append(str(path))

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0)], names=["H", "K", "L"]
        )
        result = load_datasets(files, valid_indices)

        assert result.shape == (3, 2)
        assert isinstance(result, jnp.ndarray)

    def test_dtype_parameter(self, tmp_path):
        """Test dtype parameter in load_datasets."""
        df = pd.DataFrame({
            "H": [1, 2],
            "K": [0, 0],
            "L": [0, 0],
            "f_real": [1.0, 2.0],
            "f_imag": [0.5, 1.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        valid_indices = pd.MultiIndex.from_tuples(
            [(1, 0, 0), (2, 0, 0)], names=["H", "K", "L"]
        )

        # Test complex64
        result_c64 = load_datasets([str(path)], valid_indices, dtype="complex64")
        assert result_c64.dtype == jnp.complex64

        # Test complex128 (requires x64 mode)
        import jax
        with jax.experimental.enable_x64():
            result_c128 = load_datasets([str(path)], valid_indices, dtype="complex128")
            # Check dtype is complex and has higher precision
            assert jnp.iscomplexobj(result_c128)
            assert result_c128.dtype.itemsize >= 16  # complex128 is 16 bytes

    def test_empty_file_list_raises(self):
        """Test that empty file list raises ValueError."""
        valid_indices = pd.MultiIndex.from_tuples([(1, 0, 0)], names=["H", "K", "L"])
        with pytest.raises(ValueError, match="No files loaded"):
            load_datasets([], valid_indices)


class TestDropConfigValidation:
    """Tests for DropConfig validation (from state.py)."""

    def test_valid_config(self):
        """Test that valid config works."""
        from simulation_timeseries_optim.state import DropConfig

        config = DropConfig(
            percentile_threshold=20.0,
            min_mtz_count=5,
            max_rounds=10,
        )
        assert config.percentile_threshold == 20.0

    def test_invalid_percentile_too_high(self):
        """Test that percentile > 100 raises."""
        from simulation_timeseries_optim.state import DropConfig

        with pytest.raises(ValueError, match="percentile_threshold"):
            DropConfig(percentile_threshold=101.0)

    def test_invalid_percentile_negative(self):
        """Test that negative percentile raises."""
        from simulation_timeseries_optim.state import DropConfig

        with pytest.raises(ValueError, match="percentile_threshold"):
            DropConfig(percentile_threshold=-5.0)

    def test_invalid_min_mtz_count(self):
        """Test that min_mtz_count < 1 raises."""
        from simulation_timeseries_optim.state import DropConfig

        with pytest.raises(ValueError, match="min_mtz_count"):
            DropConfig(min_mtz_count=0)

    def test_invalid_max_rounds(self):
        """Test that max_rounds < 1 raises."""
        from simulation_timeseries_optim.state import DropConfig

        with pytest.raises(ValueError, match="max_rounds"):
            DropConfig(max_rounds=0)

    def test_adaptive_dropping_validation(self):
        """Test validation of adaptive dropping parameters."""
        from simulation_timeseries_optim.state import DropConfig

        # Valid adaptive config
        config = DropConfig(adaptive_dropping=True, initial_percentile=20.0)
        assert config.adaptive_dropping

        # Invalid initial percentile
        with pytest.raises(ValueError, match="initial_percentile"):
            DropConfig(adaptive_dropping=True, initial_percentile=150.0)

        # Invalid decay rate
        with pytest.raises(ValueError, match="decay_rate"):
            DropConfig(adaptive_dropping=True, decay_rate=1.5)
