import jax.numpy as jnp
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from jaxtyping import Array, Complex, Float
from loguru import logger


def load_structure_factors(
    path: str, valid_indices: pd.Index, labels: str
) -> np.ndarray:
    """
    Load complex structure factors from either MTZ or parquet input.

    Args:
        path: Path to the file (.mtz or .parquet).
        valid_indices: Pandas Index of (H, K, L) to keep.
        labels: Comma-separated string of column names (e.g., "FMODEL,PHIFMODEL").

    Returns:
        Numpy array of complex structure factors.
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
            f_real = df["f_real"].to_numpy()
            f_imag = df["f_imag"].to_numpy()
            return f_real + f_imag * 1j

        if {amp_label, phase_label}.issubset(df.columns):
            amplitude = df[amp_label].to_numpy()
            phase = df[phase_label].to_numpy()
            if phase_label.lower().endswith("deg"):
                phase = np.deg2rad(phase)
            return amplitude * np.exp(1j * phase)

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
    )

    return dataset


def load_datasets(
    files: list[str], valid_indices: pd.Index, labels: str = "FMODEL,PHIFMODEL"
) -> Complex[Array, "n_datasets n_reflections"]:
    """
    Loads multiple datasets into a single JAX array.

    Args:
        files: List of file paths.
        valid_indices: Indices to keep.
        labels: Column labels.

    Returns:
        JAX array of shape (n_datasets, n_reflections).
    """
    datasets = []
    for f in files:
        logger.debug(f"Reading: {f.split('/')[-1]}")
        data = load_structure_factors(f, valid_indices, labels)
        datasets.append(data)

    if not datasets:
        raise ValueError("No files loaded!")

    return jnp.stack(datasets)


def load_ground_truth(
    path: str, column: str = "sqrtIdiff"
) -> tuple[Float[Array, " n_reflections"], pd.Index]:
    """
    Loads the ground truth intensity data.

    Args:
        path: Path to the MTZ file.
        column: Column name for the square root of intensity.

    Returns:
        Tuple of (intensity array, valid indices).
    """
    gt_dataset = rs.read_mtz(path).expand_to_p1()
    nas = gt_dataset.isna()
    gt_dataset = gt_dataset[~nas[column]]
    valid_indices = gt_dataset.index

    y = gt_dataset[column].to_numpy() ** 2
    return jnp.array(y), valid_indices
