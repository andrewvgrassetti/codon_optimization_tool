"""Optimization strategies for codon selection."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

from src.analysis.metrics import CodonMetricsCalculator, SequenceAnalyzer
from src.config.organisms import CodonUsageTable
from src.config.constants import AMINO_ACID_TO_CODONS


class OptimizationStrategy(ABC):
    """Abstract base class for codon optimization strategies."""

    @abstractmethod
    def select_codon(
        self, amino_acid: str, codon_table: CodonUsageTable
    ) -> str:
        """Select a codon for the given amino acid.

        Args:
            amino_acid: Single-letter amino acid code.
            codon_table: Codon usage table for the target organism.

        Returns:
            A three-letter DNA codon.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...

    def optimize_full_sequence(
        self, protein: str, codon_table: CodonUsageTable
    ) -> str | None:
        """Generate a full optimized DNA sequence in one pass.

        Override in strategies that need to generate the complete sequence at
        once (e.g. rejection-sampling strategies).  The default implementation
        returns ``None``, signalling that the optimizer should fall back to
        per-codon selection via :meth:`select_codon`.

        Args:
            protein: Amino-acid string (no stop codon).
            codon_table: Codon usage table for the target organism.

        Returns:
            An optimized DNA string, or ``None`` to use per-codon selection.
        """
        return None


def _rejection_sample(
    make_inner_strategy: Callable[[int], OptimizationStrategy],
    protein: str,
    codon_table: CodonUsageTable,
    gc_min: float | None,
    gc_max: float | None,
    wrscu_min: float | None,
    wrscu_max: float | None,
    max_attempts: int,
) -> Tuple[str, List[str]]:
    """Rejection-sampling loop shared by constrained strategies.

    Generates candidate sequences using *make_inner_strategy* (called with an
    integer seed on each attempt) and returns the first candidate that satisfies
    all supplied constraints.  If no candidate passes within *max_attempts*, the
    best candidate found (lowest total constraint violation distance) is returned
    together with a warning message.

    Args:
        make_inner_strategy: Factory that accepts an integer seed and returns a
            strategy used to generate one candidate.
        protein: Amino-acid string (no stop codon).
        codon_table: Codon usage table for the target organism.
        gc_min: Minimum acceptable GC fraction (0–1), or ``None`` to skip.
        gc_max: Maximum acceptable GC fraction (0–1), or ``None`` to skip.
        wrscu_min: Minimum acceptable wRSCU value, or ``None`` to skip.
        wrscu_max: Maximum acceptable wRSCU value, or ``None`` to skip.
        max_attempts: Maximum number of candidate sequences to try.

    Returns:
        A ``(sequence, warnings)`` tuple.  *warnings* is empty on success.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")

    best_candidate: str | None = None
    best_score = float("inf")

    for attempt in range(max_attempts):
        inner = make_inner_strategy(attempt)
        candidate = "".join(inner.select_codon(aa, codon_table) for aa in protein)

        score = 0.0
        if gc_min is not None and gc_max is not None:
            gc = SequenceAnalyzer.gc_content(candidate)
            score += max(0.0, gc_min - gc) + max(0.0, gc - gc_max)

        if wrscu_min is not None and wrscu_max is not None:
            wrscu = CodonMetricsCalculator.weighted_rscu(candidate, codon_table)
            score += max(0.0, wrscu_min - wrscu) + max(0.0, wrscu - wrscu_max)

        if score == 0.0:
            return candidate, []

        if score < best_score:
            best_score = score
            best_candidate = candidate

    warnings = [
        f"Could not satisfy all constraints after {max_attempts} attempts. "
        "Returning best candidate found."
    ]
    return best_candidate, warnings  # type: ignore[return-value]


class HighestFrequencyStrategy(OptimizationStrategy):
    """Always select the most frequently used codon (deterministic).

    This is the simplest optimization approach. It maximizes the
    Codon Adaptation Index (CAI) but may create undesirable sequence
    features like repeats or extreme GC content.
    """

    def select_codon(
        self, amino_acid: str, codon_table: CodonUsageTable
    ) -> str:
        return codon_table.get_best_codon(amino_acid)

    def name(self) -> str:
        return "Highest Frequency"


class WeightedRandomStrategy(OptimizationStrategy):
    """Select codons weighted by usage frequency.

    This avoids the repetitiveness of always choosing the top codon,
    producing more natural sequences while still favoring preferred codons.

    When *gc_min*/*gc_max* or *wrscu_min*/*wrscu_max* constraints are supplied,
    a rejection-sampling loop is used to find a sequence that satisfies them.
    If no satisfying sequence is found within *max_attempts*, the best candidate
    is returned and a warning is stored in :attr:`last_warnings`.
    When no constraints are given the strategy falls back to per-codon weighted
    random selection (original behaviour).
    """

    def __init__(
        self,
        seed: int | None = None,
        gc_min: float | None = None,
        gc_max: float | None = None,
        wrscu_min: float | None = None,
        wrscu_max: float | None = None,
        max_attempts: int = 1000,
    ) -> None:
        self._rng = random.Random(seed)
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.wrscu_min = wrscu_min
        self.wrscu_max = wrscu_max
        self.max_attempts = max_attempts
        self.last_warnings: List[str] = []

    def select_codon(
        self, amino_acid: str, codon_table: CodonUsageTable
    ) -> str:
        codon_freqs = codon_table.get_codons_for_amino_acid(amino_acid)
        if not codon_freqs:
            raise ValueError(f"No codons available for amino acid: {amino_acid}")
        codons = list(codon_freqs.keys())
        weights = list(codon_freqs.values())
        # Handle case where all weights are zero
        if sum(weights) == 0:
            return self._rng.choice(codons)
        return self._rng.choices(codons, weights=weights, k=1)[0]

    def name(self) -> str:
        return "Weighted Random"

    def optimize_full_sequence(
        self, protein: str, codon_table: CodonUsageTable
    ) -> str | None:
        """Use rejection sampling when constraints are set; otherwise return None."""
        has_gc = self.gc_min is not None and self.gc_max is not None
        has_wrscu = self.wrscu_min is not None and self.wrscu_max is not None
        if not has_gc and not has_wrscu:
            return None  # fall back to per-codon weighted random

        seq, self.last_warnings = _rejection_sample(
            lambda seed: WeightedRandomStrategy(seed=seed),
            protein,
            codon_table,
            self.gc_min,
            self.gc_max,
            self.wrscu_min,
            self.wrscu_max,
            self.max_attempts,
        )
        return seq


class RandomOptimizationStrategy(OptimizationStrategy):
    """Select codons uniformly at random (equal weight per synonymous codon).

    Unlike :class:`WeightedRandomStrategy`, all synonymous codons for an amino
    acid are equally likely regardless of their natural usage frequencies.  This
    produces high sequence diversity and can be useful when exploring the full
    codon-usage space.

    When *gc_min*/*gc_max* or *wrscu_min*/*wrscu_max* constraints are supplied,
    a rejection-sampling loop is used to find a sequence that satisfies them.
    If no satisfying sequence is found within *max_attempts*, the best candidate
    is returned and a warning is stored in :attr:`last_warnings`.
    When no constraints are given the strategy falls back to per-codon uniform
    random selection.
    """

    def __init__(
        self,
        seed: int | None = None,
        gc_min: float | None = None,
        gc_max: float | None = None,
        wrscu_min: float | None = None,
        wrscu_max: float | None = None,
        max_attempts: int = 1000,
    ) -> None:
        self._rng = random.Random(seed)
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.wrscu_min = wrscu_min
        self.wrscu_max = wrscu_max
        self.max_attempts = max_attempts
        self.last_warnings: List[str] = []

    def select_codon(
        self, amino_acid: str, codon_table: CodonUsageTable
    ) -> str:
        codon_freqs = codon_table.get_codons_for_amino_acid(amino_acid)
        if not codon_freqs:
            raise ValueError(f"No codons available for amino acid: {amino_acid}")
        codons = list(codon_freqs.keys())
        return self._rng.choice(codons)  # uniform random – equal weight

    def name(self) -> str:
        return "Random Optimization"

    def optimize_full_sequence(
        self, protein: str, codon_table: CodonUsageTable
    ) -> str | None:
        """Use rejection sampling when constraints are set; otherwise return None."""
        has_gc = self.gc_min is not None and self.gc_max is not None
        has_wrscu = self.wrscu_min is not None and self.wrscu_max is not None
        if not has_gc and not has_wrscu:
            return None  # fall back to per-codon uniform random

        seq, self.last_warnings = _rejection_sample(
            lambda seed: RandomOptimizationStrategy(seed=seed),
            protein,
            codon_table,
            self.gc_min,
            self.gc_max,
            self.wrscu_min,
            self.wrscu_max,
            self.max_attempts,
        )
        return seq


STRATEGY_REGISTRY: Dict[str, type] = {
    "highest_frequency": HighestFrequencyStrategy,
    "weighted_random": WeightedRandomStrategy,
    "random_optimization": RandomOptimizationStrategy,
}
