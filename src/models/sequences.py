"""Sequence model classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config.constants import CODON_TABLE, STOP_CODONS


@dataclass
class BaseSequence:
    """Base class for biological sequences."""

    sequence: str
    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper().strip()

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence


@dataclass
class DNASequence(BaseSequence):
    """A DNA coding sequence."""

    def translate(self) -> str:
        """Translate this DNA sequence to a protein sequence.

        Returns the amino acid string (without trailing stop).
        Raises ValueError if the sequence length is not divisible by 3.
        """
        seq = self.sequence
        if len(seq) % 3 != 0:
            raise ValueError(
                f"Sequence length ({len(seq)}) is not divisible by 3."
            )
        protein = []
        for i in range(0, len(seq), 3):
            codon = seq[i : i + 3]
            aa = CODON_TABLE.get(codon)
            if aa is None:
                raise ValueError(f"Invalid codon: {codon}")
            if aa == "*":
                break
            protein.append(aa)
        return "".join(protein)

    @property
    def gc_content(self) -> float:
        """Calculate GC content as a fraction."""
        if not self.sequence:
            return 0.0
        gc = sum(1 for base in self.sequence if base in "GC")
        return gc / len(self.sequence)

    def get_codons(self) -> list[str]:
        """Split the sequence into codons."""
        seq = self.sequence
        # Exclude trailing stop codon if present
        codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
        if codons and codons[-1] in STOP_CODONS:
            codons = codons[:-1]
        return codons


@dataclass
class ProteinSequence(BaseSequence):
    """A protein amino acid sequence."""

    pass


@dataclass
class VariantConfig:
    """Configuration for a single optimization variant.

    Each variant can have its own optimization strategy, GC content
    constraints, and wRSCU constraints while sharing other settings
    (organism, shared constraints).
    """

    strategy_name: str = "highest_frequency"
    gc_min: Optional[float] = None
    gc_max: Optional[float] = None
    wrscu_min: Optional[float] = None
    wrscu_max: Optional[float] = None

    @property
    def label(self) -> str:
        """Human-readable label describing this variant configuration."""
        strategy_label = (
            "Highest Frequency"
            if self.strategy_name == "highest_frequency"
            else "Weighted Random"
        )
        parts = [strategy_label]
        if self.gc_min is not None and self.gc_max is not None:
            parts.append(f"GC {self.gc_min:.0%}–{self.gc_max:.0%}")
        if self.wrscu_min is not None and self.wrscu_max is not None:
            parts.append(f"wRSCU {self.wrscu_min:.2f}–{self.wrscu_max:.2f}")
        return ", ".join(parts)


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    original_sequence: str
    optimized_dna: DNASequence
    protein_sequence: str
    organism_name: str
    input_type: str
    metrics_before: Optional[dict] = field(default_factory=dict)
    metrics_after: Optional[dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    variant_label: str = ""
    strategy_name: str = ""
