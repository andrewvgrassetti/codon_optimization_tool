"""Optimization constraints for codon-optimized sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.config.constants import COMMON_RESTRICTION_SITES


class OptimizationConstraint(ABC):
    """Abstract base class for post-optimization sequence constraints."""

    @abstractmethod
    def check(self, dna_sequence: str) -> List[str]:
        """Check the sequence and return a list of warning messages.

        An empty list means the constraint is satisfied.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this constraint."""
        ...


@dataclass
class GCContentConstraint(OptimizationConstraint):
    """Ensure GC content falls within a specified range."""

    min_gc: float = 0.30
    max_gc: float = 0.70

    def check(self, dna_sequence: str) -> List[str]:
        if not dna_sequence:
            return []
        gc = sum(1 for b in dna_sequence.upper() if b in "GC") / len(dna_sequence)
        warnings: List[str] = []
        if gc < self.min_gc:
            warnings.append(
                f"GC content ({gc:.1%}) is below minimum ({self.min_gc:.1%})."
            )
        if gc > self.max_gc:
            warnings.append(
                f"GC content ({gc:.1%}) is above maximum ({self.max_gc:.1%})."
            )
        return warnings

    def name(self) -> str:
        return f"GC Content ({self.min_gc:.0%}–{self.max_gc:.0%})"


@dataclass
class RestrictionSiteConstraint(OptimizationConstraint):
    """Check for the presence of restriction enzyme recognition sites."""

    sites_to_avoid: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.sites_to_avoid is None:
            self.sites_to_avoid = {}

    def check(self, dna_sequence: str) -> List[str]:
        warnings: List[str] = []
        seq = dna_sequence.upper()
        for enzyme, site in (self.sites_to_avoid or {}).items():
            if site.upper() in seq:
                pos = seq.index(site.upper()) + 1
                warnings.append(
                    f"Restriction site {enzyme} ({site}) found at position {pos}."
                )
        return warnings

    def name(self) -> str:
        return "Restriction Site Avoidance"


@dataclass
class HomopolymerConstraint(OptimizationConstraint):
    """Warn about homopolymer runs (e.g., AAAAAA)."""

    max_run_length: int = 6

    def check(self, dna_sequence: str) -> List[str]:
        warnings: List[str] = []
        seq = dna_sequence.upper()
        for base in "ATCG":
            run = base * (self.max_run_length + 1)
            if run in seq:
                pos = seq.index(run) + 1
                warnings.append(
                    f"Homopolymer run of {base} (>{self.max_run_length}bp) "
                    f"found at position {pos}."
                )
        return warnings

    def name(self) -> str:
        return f"Homopolymer (max {self.max_run_length}bp)"


@dataclass
class MotifConstraint(OptimizationConstraint):
    """Check for user-specified forbidden motifs."""

    forbidden_motifs: list[str] | None = None

    def __post_init__(self) -> None:
        if self.forbidden_motifs is None:
            self.forbidden_motifs = []

    def check(self, dna_sequence: str) -> List[str]:
        warnings: List[str] = []
        seq = dna_sequence.upper()
        for motif in self.forbidden_motifs or []:
            motif_upper = motif.upper()
            if motif_upper in seq:
                pos = seq.index(motif_upper) + 1
                warnings.append(
                    f"Forbidden motif '{motif}' found at position {pos}."
                )
        return warnings

    def name(self) -> str:
        return "Motif Avoidance"
