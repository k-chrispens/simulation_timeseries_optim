"""Visualization utilities for weight tracking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from simulation_timeseries_optim.state import OptimizationHistory


def save_weight_histogram(
    weights: np.ndarray,
    round_number: int,
    output_dir: str,
    dropped_indices: list[int] | None = None,
    percentile_threshold: float = 10.0,
) -> str:
    """
    Save a histogram of weight values with dropped MTZs highlighted.

    Args:
        weights: Weight values.
        round_number: Current round number.
        output_dir: Directory to save the plot.
        dropped_indices: Indices that will be dropped (highlighted).
        percentile_threshold: Percentile used for dropping.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate threshold
    threshold = np.percentile(weights, percentile_threshold)

    # Left: Histogram
    ax1 = axes[0]
    n_bins = min(30, len(weights))
    ax1.hist(weights, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{percentile_threshold:.0f}th percentile: {threshold:.4f}",
    )
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Round {round_number}: Weight Distribution (n={len(weights)})")
    ax1.legend()

    # Right: Bar plot of individual weights
    ax2 = axes[1]
    x = np.arange(len(weights))
    colors = [
        "red" if i in (dropped_indices or []) else "steelblue"
        for i in range(len(weights))
    ]
    ax2.bar(x, weights, color=colors, edgecolor="black", alpha=0.7, width=1.0)
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("MTZ Index")
    ax2.set_ylabel("Weight")
    ax2.set_title(f"Round {round_number}: Individual Weights")

    # Add annotation for dropped count
    n_dropped = len(dropped_indices) if dropped_indices else 0
    ax2.annotate(
        f"Dropping: {n_dropped} MTZs",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    filepath = Path(output_dir) / f"weights_round_{round_number:03d}.png"
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    return str(filepath)


def save_summary_plot(
    history: OptimizationHistory,
    output_dir: str,
) -> str:
    """
    Save a summary plot showing optimization progression across rounds.

    Args:
        history: Complete optimization history.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not history.rounds:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rounds = [r.round_number for r in history.rounds]

    # Top-left: MTZ count over rounds
    ax1 = axes[0, 0]
    mtz_counts = [r.n_mtz_start for r in history.rounds]
    mtz_counts.append(history.rounds[-1].n_mtz_end)
    round_labels = list(range(len(mtz_counts)))
    ax1.plot(
        round_labels, mtz_counts, "o-", linewidth=2, markersize=8, color="steelblue"
    )
    ax1.set_xlabel("Round (0 = initial)")
    ax1.set_ylabel("MTZ Count")
    ax1.set_title("MTZ Count Over Rounds")
    ax1.grid(True, alpha=0.3)

    # Top-right: Final loss per round
    ax2 = axes[0, 1]
    losses = [r.final_loss for r in history.rounds]
    ax2.plot(rounds, losses, "o-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Final Loss (negative CC)")
    ax2.set_title("Final Loss Per Round")
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Number dropped per round
    ax3 = axes[1, 0]
    dropped_counts = [len(r.dropped_indices) for r in history.rounds]
    ax3.bar(rounds, dropped_counts, color="coral", edgecolor="black")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("MTZs Dropped")
    ax3.set_title("MTZs Dropped Per Round")

    # Bottom-right: Full loss history concatenated
    ax4 = axes[1, 1]
    full_losses = np.concatenate([r.losses for r in history.rounds])
    ax4.plot(full_losses, linewidth=0.5, alpha=0.8)
    # Add vertical lines for round boundaries
    cumsum = 0
    for r in history.rounds[:-1]:
        cumsum += len(r.losses)
        ax4.axvline(cumsum, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Total Steps")
    ax4.set_ylabel("Loss")
    ax4.set_title("Full Loss History (red lines = round boundaries)")
    ax4.grid(True, alpha=0.3)

    # Add overall summary text
    fig.suptitle(
        f"Optimization Summary: {history.original_n_mtz} -> "
        f"{len(history.current_to_original)} MTZs over {len(history.rounds)} rounds",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    filepath = Path(output_dir) / "optimization_summary.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(filepath)


def save_final_weights_plot(
    weights: np.ndarray,
    mtz_files: list[str],
    output_dir: str,
) -> str:
    """
    Save a plot of the final retained weights with file labels.

    Args:
        weights: Final weight values.
        mtz_files: List of retained MTZ file paths.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(weights) * 0.3)))

    # Sort by weight for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_files = [Path(mtz_files[i]).stem for i in sorted_indices]

    y_pos = np.arange(len(sorted_weights))
    ax.barh(y_pos, sorted_weights, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_files, fontsize=8)
    ax.set_xlabel("Weight")
    ax.set_title("Final Retained Weights (sorted)")
    ax.invert_yaxis()

    plt.tight_layout()

    filepath = Path(output_dir) / "final_weights.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(filepath)
