"""Core codon optimizer engine."""

from __future__ import annotations

from typing import List, Optional

from src.config.constants import CODON_TABLE, STOP_CODONS
from src.config.organisms import CodonUsageTable, OrganismProfile
from src.models.sequences import DNASequence
from src.optimization.constraints import OptimizationConstraint
from src.optimization.strategies import (
    HighestFrequencyStrategy,
    OptimizationStrategy,
)


class CodonOptimizer:
    """Codon optimization engine.

    Takes a protein sequence and produces an optimized DNA coding sequence
    for a given target organism using a pluggable strategy.
    """

    def __init__(
        self,
        organism: OrganismProfile,
        strategy: OptimizationStrategy | None = None,
        constraints: List[OptimizationConstraint] | None = None,
        add_stop_codon: bool = True,
    ) -> None:
        self.organism = organism
        self.strategy = strategy or HighestFrequencyStrategy()
        self.constraints = constraints or []
        self.add_stop_codon = add_stop_codon

    def optimize_from_protein(self, protein_sequence: str) -> DNASequence:
        """Generate an optimized DNA sequence from a protein sequence.

        Args:
            protein_sequence: Amino acid string (single-letter codes, no stop).

        Returns:
            An optimized DNASequence.
        """
        protein = protein_sequence.upper().strip().rstrip("*")
        codons: List[str] = []

        for aa in protein:
            codon = self.strategy.select_codon(aa, self.organism.codon_table)
            codons.append(codon)

        if self.add_stop_codon:
            # Use TAA as the default stop codon (most common in many organisms)
            codons.append("TAA")

        dna_str = "".join(codons)
        return DNASequence(sequence=dna_str)

    def optimize_from_dna(self, dna_sequence: str) -> DNASequence:
        """Optimize an existing DNA coding sequence.

        Translates to protein, then back-translates with optimized codons.

        Args:
            dna_sequence: DNA coding sequence string.

        Returns:
            An optimized DNASequence preserving the amino acid sequence.
        """
        source = DNASequence(sequence=dna_sequence)
        protein = source.translate()
        has_stop = dna_sequence.upper().strip()
        last_codon = has_stop[-3:] if len(has_stop) >= 3 else ""
        original_has_stop = last_codon in STOP_CODONS

        # Temporarily set stop codon preference based on original
        old_stop = self.add_stop_codon
        self.add_stop_codon = original_has_stop
        result = self.optimize_from_protein(protein)
        self.add_stop_codon = old_stop

        return result

    def check_constraints(self, dna_sequence: str) -> List[str]:
        """Run all constraints against a DNA sequence.

        Returns:
            List of warning messages from all constraints.
        """
        all_warnings: List[str] = []
        for constraint in self.constraints:
            all_warnings.extend(constraint.check(dna_sequence))
        return all_warnings
