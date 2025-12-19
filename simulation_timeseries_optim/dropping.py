"""MTZ dropping logic for iterative weight optimization."""

import numpy as np
from jaxtyping import Array, Complex, Float
from loguru import logger

from simulation_timeseries_optim.models import Weights
from simulation_timeseries_optim.state import (
    DropConfig,
    OptimizationHistory,
    RoundResult,
)
from simulation_timeseries_optim.train import train


def identify_mtz_to_drop(
    weights: Float[Array, " n_mtz"],
    config: DropConfig,
    n_remaining: int,
) -> list[int]:
    """
    Identify which MTZ indices should be dropped based on weight values.

    Args:
        weights: Current weight values for each MTZ.
        config: Dropping configuration.
        n_remaining: Number of MTZs currently remaining.

    Returns:
        List of indices to drop (relative to current array).
    """
    weights_np = np.asarray(weights)

    # Calculate percentile threshold
    percentile_value = np.percentile(weights_np, config.percentile_threshold)

    # Also apply absolute minimum if set
    effective_threshold = max(percentile_value, config.min_weight_threshold)

    # Find indices below threshold
    drop_mask = weights_np < effective_threshold
    drop_indices = np.where(drop_mask)[0].tolist()

    # Ensure we don't drop too many (keep at least min_mtz_count)
    remaining_after_drop = n_remaining - len(drop_indices)
    if remaining_after_drop < config.min_mtz_count:
        # Sort by weight, only drop enough to reach min_mtz_count
        sorted_indices = np.argsort(weights_np)
        max_to_drop = max(0, n_remaining - config.min_mtz_count)
        drop_indices = sorted_indices[:max_to_drop].tolist()

    return drop_indices


def drop_mtzs(
    F_array: Complex[Array, "time hkl"],
    indices_to_drop: list[int],
) -> Complex[Array, "time_reduced hkl"]:
    """
    Remove specified MTZ datasets from the F_array.

    Args:
        F_array: Current structure factor array.
        indices_to_drop: Indices to remove.

    Returns:
        Reduced F_array with dropped MTZs removed.
    """
    if not indices_to_drop:
        return F_array

    keep_mask = np.ones(F_array.shape[0], dtype=bool)
    keep_mask[list(indices_to_drop)] = False

    return F_array[keep_mask]


def run_single_round(
    F_array: Complex[Array, "time hkl"],
    y: Float[Array, " hkl"],
    round_number: int,
    n_steps: int,
    learning_rate: float,
    lambda_l1: float,
    lambda_l2: float,
    use_proximal: bool,
    method: str,
    config: DropConfig,
    current_to_original: np.ndarray,
) -> tuple[RoundResult, Complex[Array, "time_reduced hkl"]]:
    """
    Run a single optimization round and identify MTZs to drop.

    Args:
        F_array: Current structure factor array.
        y: Ground truth intensity.
        round_number: Current round number.
        n_steps: Optimization steps per round.
        learning_rate: Learning rate for optimizer.
        lambda_l1: L1 regularization strength.
        lambda_l2: L2 regularization strength.
        use_proximal: Whether to use proximal gradient.
        method: Weight parameterization method.
        config: Dropping configuration.
        current_to_original: Mapping from current to original indices.

    Returns:
        Tuple of (round_result, reduced_F_array).
    """
    n_mtz = F_array.shape[0]

    # Initialize fresh model for this round
    model = Weights(n_timepoints=n_mtz, method=method)

    logger.info(f"Round {round_number}: Starting with {n_mtz} MTZs")

    # Run training
    final_model, losses = train(
        model=model,
        F_array=F_array,
        y=y,
        n_steps=n_steps,
        learning_rate=learning_rate,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        use_proximal=use_proximal,
    )

    # Get final weights
    final_weights = final_model()

    # Identify MTZs to drop
    local_drop_indices = identify_mtz_to_drop(final_weights, config, n_mtz)

    # Map local indices to original indices
    original_drop_indices = [int(current_to_original[i]) for i in local_drop_indices]

    # Create round result
    result = RoundResult(
        round_number=round_number,
        n_mtz_start=n_mtz,
        n_mtz_end=n_mtz - len(local_drop_indices),
        dropped_indices=local_drop_indices,
        dropped_original_indices=original_drop_indices,
        final_weights=np.asarray(final_weights),
        final_loss=float(losses[-1]),
        losses=np.asarray(losses),
    )

    logger.info(f"Round {round_number}: Final loss = {result.final_loss:.6f}")
    if local_drop_indices:
        logger.info(
            f"Round {round_number}: Dropping {len(local_drop_indices)} MTZs "
            f"(original indices: {original_drop_indices})"
        )
    else:
        logger.info(f"Round {round_number}: No MTZs to drop")

    # Drop MTZs for next round
    reduced_F_array = drop_mtzs(F_array, local_drop_indices)

    return result, reduced_F_array


def run_iterative_optimization(
    F_array: Complex[Array, "time hkl"],
    y: Float[Array, " hkl"],
    mtz_files: list[str],
    n_steps: int,
    learning_rate: float,
    lambda_l1: float,
    lambda_l2: float,
    use_proximal: bool,
    method: str,
    config: DropConfig,
    output_dir: str | None = None,
) -> OptimizationHistory:
    """
    Run the full iterative optimization with MTZ dropping.

    Args:
        F_array: Initial structure factor array.
        y: Ground truth intensity.
        mtz_files: List of original MTZ file paths.
        n_steps: Optimization steps per round.
        learning_rate: Learning rate.
        lambda_l1: L1 regularization.
        lambda_l2: L2 regularization.
        use_proximal: Use proximal gradient.
        method: Weight parameterization method.
        config: Dropping configuration.
        output_dir: Directory for output plots (optional).

    Returns:
        OptimizationHistory with all round results.
    """
    history = OptimizationHistory(
        original_n_mtz=F_array.shape[0],
        original_mtz_files=mtz_files,
    )

    current_F_array = F_array
    round_number = 0

    logger.info("=" * 60)
    logger.info("ITERATIVE MTZ DROPPING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Starting with {history.original_n_mtz} MTZs")
    logger.info(f"Drop percentile: {config.percentile_threshold}%")
    logger.info(f"Min MTZ count: {config.min_mtz_count}")
    logger.info(f"Max rounds: {config.max_rounds}")

    while True:
        round_number += 1

        # Check stopping conditions
        if round_number > config.max_rounds:
            logger.info(f"Stopping: Reached max rounds ({config.max_rounds})")
            break

        current_n_mtz = current_F_array.shape[0]
        if current_n_mtz <= config.min_mtz_count:
            logger.info(
                f"Stopping: Reached minimum MTZ count ({config.min_mtz_count})"
            )
            break

        # Run single round
        result, reduced_F_array = run_single_round(
            F_array=current_F_array,
            y=y,
            round_number=round_number,
            n_steps=n_steps,
            learning_rate=learning_rate,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            use_proximal=use_proximal,
            method=method,
            config=config,
            current_to_original=history.current_to_original,
        )

        # Save visualization if output_dir provided
        if output_dir:
            from simulation_timeseries_optim.visualization import save_weight_histogram

            save_weight_histogram(
                weights=result.final_weights,
                round_number=round_number,
                output_dir=output_dir,
                dropped_indices=result.dropped_indices,
                percentile_threshold=config.percentile_threshold,
            )

        # Update history (this also updates current_to_original mapping)
        history.add_round(result)

        # Check if no MTZs were dropped
        if len(result.dropped_indices) == 0:
            logger.info("Stopping: No MTZs dropped in this round")
            break

        current_F_array = reduced_F_array

    # Save summary plot
    if output_dir:
        from simulation_timeseries_optim.visualization import save_summary_plot

        save_summary_plot(history, output_dir)

    logger.info("=" * 60)
    logger.info("ITERATIVE OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rounds: {len(history.rounds)}")
    logger.info(
        f"MTZs retained: {len(history.current_to_original)} / {history.original_n_mtz}"
    )
    logger.info(f"Total dropped: {history.get_total_dropped()}")

    return history
