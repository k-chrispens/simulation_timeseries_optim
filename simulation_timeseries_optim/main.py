import argparse
import glob

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from loguru import logger

from simulation_timeseries_optim.dropping import run_iterative_optimization
from simulation_timeseries_optim.io import (
    load_datasets,
    load_datasets_sharded,
    load_ground_truth,
)
from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.state import DropConfig
from simulation_timeseries_optim.train import train
from simulation_timeseries_optim.visualization import save_final_weights_plot


def main():
    parser = argparse.ArgumentParser(
        description="Optimize weights for diffuse scattering."
    )
    parser.add_argument(
        "--mtz-dir", type=str, default="mtz1.8A", help="Directory containing MTZ files"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="sqrtIdiff_observed.mtz",
        help="Ground truth MTZ file",
    )
    parser.add_argument(
        "--n-steps", type=int, default=500, help="Number of optimization steps"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument(
        "--l1", type=float, default=0.1, help="L1 regularization strength"
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 regularization strength"
    )
    parser.add_argument(
        "--proximal", action="store_true", help="Use proximal gradient descent"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="raw",
        choices=["raw", "softmax", "sigmoid"],
        help="Parameterization method",
    )

    # Iterative dropping arguments
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative MTZ dropping mode",
    )
    parser.add_argument(
        "--drop-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for dropping (drop bottom X%%)",
    )
    parser.add_argument(
        "--min-mtz",
        type=int,
        default=50,
        help="Minimum number of MTZs to retain",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum number of dropping iterations",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Absolute minimum weight threshold for dropping",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files and plots",
    )

    # dtype and performance arguments
    parser.add_argument(
        "--dtype",
        type=str,
        default="complex64",
        choices=["complex64", "complex128"],
        help="Data type for structure factors (complex64 saves 50%% memory)",
    )
    parser.add_argument(
        "--float-dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type for weights and intensities",
    )
    parser.add_argument(
        "--use-sharding",
        action="store_true",
        help="Enable multi-GPU sharding for large datasets",
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        default=None,
        help="Number of GPUs to use (None = all available)",
    )
    parser.add_argument(
        "--use-remat",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory (may be slower)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of reflections to keep (0.0 to 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )

    # Early stopping and adaptive dropping arguments
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=1e-4,
        help="Stop if loss improvement is below this threshold",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Number of rounds without improvement before early stopping",
    )
    parser.add_argument(
        "--adaptive-dropping",
        action="store_true",
        help="Use adaptive dropping rate that decreases over rounds",
    )
    parser.add_argument(
        "--initial-percentile",
        type=float,
        default=20.0,
        help="Initial percentile for adaptive dropping",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.8,
        help="Decay rate for adaptive dropping percentile",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DIFFUSE SCATTERING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args}")

    # Load Ground Truth
    logger.info(f"Loading ground truth from {args.ground_truth}...")
    y, valid_indices = load_ground_truth(args.ground_truth, dtype=args.float_dtype)
    logger.info(f"Ground truth shape: {y.shape}")

    # Load Datasets
    logger.info(f"Loading datasets from {args.mtz_dir}...")
    # Support both directory of MTZs and single Parquet
    if args.mtz_dir.endswith(".parquet"):
        files = [args.mtz_dir]
    else:
        files = sorted(glob.glob(f"{args.mtz_dir}/*.mtz"))
        # Filter out ground truth if in same dir
        files = [f for f in files if "sqrtIdiffuse" not in f]

    if not files:
        logger.error("No files found!")
        return

    # Load datasets with optional sharding
    mesh: Mesh | None = None
    if args.use_sharding:
        # Load with sharding and subsampling
        result_sharded = load_datasets_sharded(
            files,
            valid_indices,
            labels="FC,PHIC",
            dtype=args.dtype,
            n_devices=args.n_devices,
            subsample_fraction=args.subsample,
            random_seed=args.random_seed,
            y=np.asarray(y),
        )

        # Unpack result based on length
        if len(result_sharded) == 3:
            F_array, y, mesh = result_sharded
        else:
            F_array, mesh = result_sharded

        n_dev = len(mesh.devices)
        logger.info(
            f"F_array shape: {F_array.shape} (sharded across {n_dev} devices)"
        )
    else:
        # Load datasets with subsampling
        result = load_datasets(
            files,
            valid_indices,
            labels="FC,PHIC",
            dtype=args.dtype,
            subsample_fraction=args.subsample,
            random_seed=args.random_seed,
            y=np.asarray(y),
        )

        # Unpack result
        if isinstance(result, tuple):
            F_array, y = result
        else:
            F_array = result

        logger.info(f"F_array shape: {F_array.shape}")

    if args.iterative:
        # Iterative MTZ dropping mode
        config = DropConfig(
            percentile_threshold=args.drop_percentile,
            min_mtz_count=args.min_mtz,
            max_rounds=args.max_rounds,
            min_weight_threshold=args.min_weight,
            early_stop_loss_threshold=args.early_stop_threshold,
            early_stop_patience=args.early_stop_patience,
            adaptive_dropping=args.adaptive_dropping,
            initial_percentile=args.initial_percentile,
            decay_rate=args.decay_rate,
        )

        history = run_iterative_optimization(
            F_array=F_array,
            y=y,
            mtz_files=files,
            n_steps=args.n_steps,
            learning_rate=args.lr,
            lambda_l1=args.l1,
            lambda_l2=args.l2,
            use_proximal=args.proximal,
            use_remat=args.use_remat,
            method=args.method,
            config=config,
            output_dir=args.output_dir,
        )

        # Save final weights plot
        final_weights = history.get_final_weights()
        if final_weights is not None:
            active_files = history.get_active_files()
            save_final_weights_plot(final_weights, active_files, args.output_dir)

        # Report retained files
        logger.info("Retained MTZ files:")
        for f in history.get_active_files():
            logger.info(f"  {f}")

    else:
        # Original single-run mode
        n_timepoints = F_array.shape[0]
        model = Weights(n_timepoints, method=args.method)

        logger.info("Starting optimization...")
        final_model, losses = train(
            model,
            F_array,
            y,
            n_steps=args.n_steps,
            learning_rate=args.lr,
            lambda_l1=args.l1,
            lambda_l2=args.l2,
            use_proximal=args.proximal,
            use_remat=args.use_remat,
        )

        # Results
        final_weights = final_model()
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Final Loss: {losses[-1]:.6f}")

        logger.info("Weight statistics:")
        logger.info(f"  Mean: {jnp.mean(final_weights):.6f}")
        logger.info(f"  Max: {jnp.max(final_weights):.6f}")
        logger.info(
            f"  Sparsity (exact 0): {jnp.sum(final_weights == 0)}/{n_timepoints}"
        )
        logger.info(
            f"  Sparsity (< 0.01): {jnp.sum(final_weights < 0.01)}/{n_timepoints}"
        )


if __name__ == "__main__":
    main()
