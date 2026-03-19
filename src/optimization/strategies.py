"""Optimization strategies for codon selection."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List

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
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

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


STRATEGY_REGISTRY: Dict[str, type] = {
    "highest_frequency": HighestFrequencyStrategy,
    "weighted_random": WeightedRandomStrategy,
}
