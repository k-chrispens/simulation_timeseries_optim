from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Complex, Float
from loguru import logger


def load_structure_factors(
    path: str,
    valid_indices: pd.Index,
    labels: str,
    dtype: np.dtype[Any] | type[np.complexfloating[Any, Any]] = np.complex64,
) -> np.ndarray:
    """
    Load complex structure factors from either MTZ or parquet input.

    Args:
        path: Path to the file (.mtz or .parquet).
        valid_indices: Pandas Index of (H, K, L) to keep.
        labels: Comma-separated string of column names (e.g., "FMODEL,PHIFMODEL").
        dtype: NumPy dtype for the output array (default: complex64 for memory
            efficiency).

    Returns:
        Numpy array of complex structure factors with specified dtype.
    """
    amp_label, phase_label = [label.strip() for label in labels.split(",")]

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        missing_columns = {"H", "K", "L"} - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Parquet file {path} missing expected Miller index columns: "
                f"{missing_columns}"
            )

        df = df.set_index(["H", "K", "L"])
        try:
            df = df.loc[valid_indices]
        except KeyError as exc:
            missing = valid_indices.difference(df.index)
            sample_missing = list(missing[:5]) if hasattr(missing, "__iter__") else []
            raise KeyError(
                f"Parquet file {path} does not contain all required reflections; "
                f"example missing entries: {sample_missing}"
            ) from exc

        if {"f_real", "f_imag"}.issubset(df.columns):
            # Convert to appropriate float dtype, then combine to complex
            float_dtype = np.float32 if dtype == np.complex64 else np.float64
            f_real = df["f_real"].to_numpy().astype(float_dtype)
            f_imag = df["f_imag"].to_numpy().astype(float_dtype)
            return (f_real + f_imag * 1j).astype(dtype)

        if {amp_label, phase_label}.issubset(df.columns):
            float_dtype = np.float32 if dtype == np.complex64 else np.float64
            amplitude = df[amp_label].to_numpy().astype(float_dtype)
            phase = df[phase_label].to_numpy().astype(float_dtype)
            if phase_label.lower().endswith("deg"):
                phase = np.deg2rad(phase)
            return (amplitude * np.exp(1j * phase)).astype(dtype)

        raise ValueError(
            f"Unable to locate structure factor columns in {path}; available columns: "
            f"{sorted(df.columns)}"
        )

    dataset = (
        rs.read_mtz(path)
        .expand_to_p1()
        .loc[valid_indices]
        .to_structurefactor(amp_label, phase_label)
        .to_numpy()
        .astype(dtype)
    )

    return dataset


def subsample_reflections(
    datasets: list[np.ndarray] | list,
    y: np.ndarray | jnp.ndarray,
    subsample_fraction: float = 1.0,
    random_seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Subsample reflections to reduce memory usage.

    Args:
        datasets: List of numpy arrays (one per dataset).
        y: Ground truth intensities.
        subsample_fraction: Fraction of reflections to keep (0.0 to 1.0).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (subsampled_datasets, subsampled_y).
    """
    if subsample_fraction >= 1.0:
        return datasets, np.asarray(y)

    # Convert to numpy if needed
    y_np = np.asarray(y)
    datasets_np = [np.asarray(d) for d in datasets]

    n_reflections = y_np.shape[0]
    n_keep = int(n_reflections * subsample_fraction)

    logger.info(
        f"Subsampling reflections: {n_reflections} -> {n_keep} "
        f"({subsample_fraction * 100:.1f}%)"
    )

    # Use numpy for reproducible random sampling
    rng = np.random.RandomState(random_seed)
    indices = rng.choice(n_reflections, size=n_keep, replace=False)
    indices = np.sort(indices)  # Keep sorted for better memory access

    subsampled_datasets = [d[indices] for d in datasets_np]
    subsampled_y = y_np[indices]

    return subsampled_datasets, subsampled_y


def load_datasets(
    files: list[str],
    valid_indices: pd.Index,
    labels: str = "FMODEL,PHIFMODEL",
    dtype: str = "complex64",
    subsample_fraction: float = 1.0,
    random_seed: int = 42,
    y: np.ndarray | jnp.ndarray | None = None,
) -> (
    Complex[Array, "n_datasets n_reflections"]
    | tuple[Complex[Array, "n_datasets n_reflections"], Float[Array, " n_reflections"]]
):
    """
    Loads multiple datasets into a single JAX array.

    Args:
        files: List of file paths.
        valid_indices: Indices to keep.
        labels: Column labels.
        dtype: Data type string ("complex64" or "complex128"). Default: complex64.
        subsample_fraction: Fraction of reflections to keep (0.0 to 1.0).
        random_seed: Random seed for subsampling.
        y: Optional ground truth array to subsample alongside datasets.

    Returns:
        If y is provided: Tuple of (F_array, y_subsampled).
        Otherwise: Just F_array.
    """
    np_dtype = np.complex64 if dtype == "complex64" else np.complex128

    datasets = []
    for f in files:
        logger.debug(f"Reading: {f.split('/')[-1]}")
        data = load_structure_factors(f, valid_indices, labels, dtype=np_dtype)
        datasets.append(data)

    if not datasets:
        raise ValueError("No files loaded!")

    # Subsample if requested and y is provided
    if subsample_fraction < 1.0 and y is not None:
        y_np = np.asarray(y)
        datasets, y_np = subsample_reflections(
            datasets, y_np, subsample_fraction, random_seed
        )
        # Stack in NumPy first, then convert to JAX with dtype preserved
        stacked = np.stack(datasets)
        float_dtype = np.float32 if dtype == "complex64" else np.float64
        return (
            jnp.array(stacked, dtype=np_dtype),
            jnp.array(y_np, dtype=float_dtype),
        )

    # Stack in NumPy first, then convert to JAX with dtype preserved
    stacked = np.stack(datasets)
    F_array = jnp.array(stacked, dtype=np_dtype)

    if y is not None:
        return F_array, jnp.asarray(y)
    return F_array


def load_ground_truth(
    path: str, column: str = "sqrtIdiff", dtype: str = "float32"
) -> tuple[Float[Array, " n_reflections"], pd.Index]:
    """
    Loads the ground truth intensity data.

    Args:
        path: Path to the MTZ file.
        column: Column name for the square root of intensity.
        dtype: Data type string ("float32" or "float64"). Default: float32.

    Returns:
        Tuple of (intensity array, valid indices).
    """
    np_dtype = np.float32 if dtype == "float32" else np.float64

    gt_dataset = rs.read_mtz(path).expand_to_p1()
    nas = gt_dataset.isna()
    gt_dataset = gt_dataset[~nas[column]]
    valid_indices = gt_dataset.index

    y = gt_dataset[column].to_numpy().astype(np_dtype) ** 2
    return jnp.array(y), valid_indices


def get_mesh_sharding(n_devices: int | None = None) -> Mesh:
    """
    Create a mesh for sharding across available GPUs.

    Args:
        n_devices: Number of devices to use (None = use all available).

    Returns:
        Mesh object for sharding across devices.
    """
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        logger.warning("No GPU devices found, falling back to CPU")
        devices = jax.devices("cpu")

    if n_devices:
        devices = devices[:n_devices]

    logger.info(f"Using {len(devices)} device(s): {[d.device_kind for d in devices]}")

    mesh_devices = mesh_utils.create_device_mesh((len(devices),))
    mesh = Mesh(mesh_devices, axis_names=("data",))

    return mesh


def load_datasets_sharded(
    files: list[str],
    valid_indices: pd.Index,
    labels: str = "FMODEL,PHIFMODEL",
    dtype: str = "complex64",
    n_devices: int | None = None,
    subsample_fraction: float = 1.0,
    random_seed: int = 42,
    y: np.ndarray | jnp.ndarray | None = None,
) -> (
    tuple[Complex[Array, "n_datasets n_reflections"], Mesh]
    | tuple[
        Complex[Array, "n_datasets n_reflections"], Float[Array, " n_reflections"], Mesh
    ]
):
    """
    Load datasets with multi-GPU sharding for large-scale computation.

    Data is sharded along the dataset (MTZ) dimension, distributing different
    MTZ files across different GPUs for parallel computation.

    Args:
        files: List of file paths.
        valid_indices: Indices to keep.
        labels: Column labels.
        dtype: Data type string ("complex64" or "complex128").
        n_devices: Number of devices to use (None = all available).
        subsample_fraction: Fraction of reflections to keep (0.0 to 1.0).
        random_seed: Random seed for subsampling.
        y: Optional ground truth array to subsample alongside datasets.

    Returns:
        If y provided: Tuple of (sharded F_array, y, mesh object).
        Otherwise: Tuple of (sharded F_array, mesh object).
    """
    # Load datasets normally first (with subsampling if requested)
    y_np = np.asarray(y) if y is not None else None
    result = load_datasets(
        files, valid_indices, labels, dtype, subsample_fraction, random_seed, y_np
    )

    # Unpack result
    if isinstance(result, tuple):
        F_array, y_result = result
    else:
        F_array = result
        y_result = None

    # Create mesh and sharding
    mesh = get_mesh_sharding(n_devices)
    sharding = NamedSharding(mesh, PartitionSpec("data", None))

    # Shard the array across devices
    F_array_sharded = jax.device_put(F_array, sharding)

    logger.info(f"Data sharded across devices: {F_array_sharded.sharding}")
    n_files_per_device = F_array_sharded.shape[0] // len(mesh.devices)
    logger.info(f"Shape per device: {n_files_per_device} MTZ files")

    if y_result is not None:
        return F_array_sharded, y_result, mesh
    return F_array_sharded, mesh
