"""State tracking dataclasses for iterative MTZ dropping optimization."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DropConfig:
    """Configuration for the MTZ dropping strategy."""

    percentile_threshold: float = 10.0  # Drop bottom X%
    min_mtz_count: int = 50  # Stop if fewer than this remain
    max_rounds: int = 10  # Maximum dropping iterations
    min_weight_threshold: float = 0.0  # Absolute minimum weight threshold (optional)


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
    current_to_original: np.ndarray = field(default=None)

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
