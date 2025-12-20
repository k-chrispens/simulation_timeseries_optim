"""State tracking dataclasses for iterative MTZ dropping optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DropConfig:
    """Configuration for the MTZ dropping strategy."""

    percentile_threshold: float = 10.0  # Drop bottom X%
    min_mtz_count: int = 50  # Stop if fewer than this remain
    max_rounds: int = 10  # Maximum dropping iterations
    min_weight_threshold: float = 0.0  # Absolute minimum weight threshold (optional)

    # Early stopping criteria
    early_stop_loss_threshold: float = 1e-4  # Stop if loss improvement < this
    early_stop_patience: int = 2  # Rounds without improvement before stopping

    # Adaptive dropping
    adaptive_dropping: bool = False
    initial_percentile: float = 20.0  # Start aggressive
    min_percentile: float = 5.0  # End conservative
    decay_rate: float = 0.8  # Multiply percentile by this each round

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.percentile_threshold <= 100.0:
            raise ValueError(
                f"percentile_threshold must be between 0 and 100, "
                f"got {self.percentile_threshold}"
            )
        if self.min_mtz_count < 1:
            raise ValueError(
                f"min_mtz_count must be at least 1, got {self.min_mtz_count}"
            )
        if self.max_rounds < 1:
            raise ValueError(
                f"max_rounds must be at least 1, got {self.max_rounds}"
            )
        if self.min_weight_threshold < 0.0:
            raise ValueError(
                f"min_weight_threshold must be non-negative, "
                f"got {self.min_weight_threshold}"
            )
        if self.early_stop_loss_threshold < 0.0:
            raise ValueError(
                f"early_stop_loss_threshold must be non-negative, "
                f"got {self.early_stop_loss_threshold}"
            )
        if self.early_stop_patience < 1:
            raise ValueError(
                f"early_stop_patience must be at least 1, "
                f"got {self.early_stop_patience}"
            )
        if self.adaptive_dropping:
            if not 0.0 <= self.initial_percentile <= 100.0:
                raise ValueError(
                    f"initial_percentile must be between 0 and 100, "
                    f"got {self.initial_percentile}"
                )
            if not 0.0 <= self.min_percentile <= 100.0:
                raise ValueError(
                    f"min_percentile must be between 0 and 100, "
                    f"got {self.min_percentile}"
                )
            if self.decay_rate <= 0.0 or self.decay_rate >= 1.0:
                raise ValueError(
                    f"decay_rate must be between 0 and 1, got {self.decay_rate}"
                )


@dataclass
class RoundResult:
    """Results from a single optimization round."""

    round_number: int
    n_mtz_start: int
    n_mtz_end: int
    dropped_indices: list[int]  # Local indices that were dropped
    dropped_original_indices: list[int]  # Original MTZ indices that were dropped
    final_weights: np.ndarray  # Weights at end of round
    final_loss: float
    losses: np.ndarray  # Full loss history for this round


@dataclass
class OptimizationHistory:
    """Tracks the full iterative optimization process."""

    original_n_mtz: int
    original_mtz_files: list[str]
    rounds: list[RoundResult] = field(default_factory=list)
    current_to_original: np.ndarray = field(default=np.ndarray([]))

    def __post_init__(self):
        if self.current_to_original is None:
            self.current_to_original = np.arange(self.original_n_mtz)

    def get_active_files(self) -> list[str]:
        """Get currently active MTZ files."""
        return [self.original_mtz_files[i] for i in self.current_to_original]

    def get_active_indices(self) -> list[int]:
        """Get currently active original MTZ indices."""
        return self.current_to_original.tolist()

    def add_round(self, result: RoundResult):
        """Add a round result and update index mapping."""
        self.rounds.append(result)
        # Remove dropped indices from mapping
        if result.dropped_indices:
            mask = np.ones(len(self.current_to_original), dtype=bool)
            for idx in result.dropped_indices:
                mask[idx] = False
            self.current_to_original = self.current_to_original[mask]

    def get_final_weights(self) -> np.ndarray | None:
        """Get the final weights from the last round."""
        if self.rounds:
            return self.rounds[-1].final_weights
        return None

    def get_total_dropped(self) -> int:
        """Get total number of MTZs dropped across all rounds."""
        return sum(len(r.dropped_indices) for r in self.rounds)
