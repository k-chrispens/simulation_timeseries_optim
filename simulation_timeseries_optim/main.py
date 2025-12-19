import argparse
import glob

import jax.numpy as jnp
from loguru import logger

from simulation_timeseries_optim.io import load_datasets, load_ground_truth
from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.train import train


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

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DIFFUSE SCATTERING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args}")

    # Load Ground Truth
    logger.info(f"Loading ground truth from {args.ground_truth}...")
    y, valid_indices = load_ground_truth(args.ground_truth)
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

    # For demonstration, we load all. In production, might want to batch or shard.
    # The original code handled sharding. JAX handles sharding automatically if
    # arrays are sharded.
    # Here we load into memory.
    F_array = load_datasets(files, valid_indices, labels="FC,PHIC")
    logger.info(f"F_array shape: {F_array.shape}")

    # Initialize Model
    n_timepoints = F_array.shape[0]
    model = Weights(n_timepoints, method=args.method)

    # Train
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
    logger.info(f"  Sparsity (exact 0): {jnp.sum(final_weights == 0)}/{n_timepoints}")
    logger.info(f"  Sparsity (< 0.01): {jnp.sum(final_weights < 0.01)}/{n_timepoints}")


if __name__ == "__main__":
    main()
