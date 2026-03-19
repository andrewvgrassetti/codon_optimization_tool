"""High-level optimization service orchestrating the full workflow."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.analysis.metrics import CodonMetricsCalculator
from src.config.organisms import OrganismProfile, OrganismRegistry, get_default_registry
from src.models.sequences import DNASequence, OptimizationResult, ProteinSequence
from src.optimization.constraints import (
    GCContentConstraint,
    HomopolymerConstraint,
    MotifConstraint,
    OptimizationConstraint,
    RestrictionSiteConstraint,
)
from src.optimization.optimizer import CodonOptimizer
from src.optimization.strategies import (
    HighestFrequencyStrategy,
    OptimizationStrategy,
    WeightedRandomStrategy,
)
from src.validation.validators import SequenceValidator, ValidationResult


class OptimizationService:
    """Orchestrates the complete optimization workflow.

    This is the main entry point for the UI layer. It coordinates validation,
    optimization, constraint checking, and metrics computation.
    """

    def __init__(self, registry: OrganismRegistry | None = None) -> None:
        self.registry = registry or get_default_registry()
        self.calculator = CodonMetricsCalculator()

    def get_organisms(self) -> List[OrganismProfile]:
        """Return available target organisms."""
        return self.registry.list_organisms()

    def validate_sequence(
        self, sequence: str, input_type: str
    ) -> ValidationResult:
        """Validate a sequence based on its type.

        Args:
            sequence: The raw sequence string.
            input_type: Either 'dna' or 'protein'.
        """
        if input_type == "dna":
            return SequenceValidator.validate_dna(sequence)
        elif input_type == "protein":
            return SequenceValidator.validate_protein(sequence)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

    def optimize(
        self,
        sequence: str,
        input_type: str,
        organism_name: str,
        strategy_name: str = "highest_frequency",
        constraints: List[OptimizationConstraint] | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run the full optimization pipeline.

        Args:
            sequence: Raw sequence string (DNA or protein).
            input_type: 'dna' or 'protein'.
            organism_name: Internal name of the target organism.
            strategy_name: Name of the optimization strategy.
            constraints: List of constraint objects.
            seed: Optional random seed for reproducibility.

        Returns:
            OptimizationResult with all relevant data.
        """
        organism = self.registry.get(organism_name)
        if organism is None:
            raise ValueError(f"Unknown organism: {organism_name}")

        # Build strategy
        strategy = self._build_strategy(strategy_name, seed)

        # Build optimizer
        optimizer = CodonOptimizer(
            organism=organism,
            strategy=strategy,
            constraints=constraints or [],
        )

        # Clean sequence
        clean_seq = sequence.upper().strip()

        # Determine protein and compute pre-optimization metrics
        metrics_before: Dict[str, object] = {}
        if input_type == "dna":
            dna_in = DNASequence(sequence=clean_seq)
            protein_seq = dna_in.translate()
            metrics_before = self.calculator.compute_metrics(
                clean_seq, organism.codon_table
            )
            optimized = optimizer.optimize_from_dna(clean_seq)
        elif input_type == "protein":
            protein_seq = clean_seq.rstrip("*")
            optimized = optimizer.optimize_from_protein(protein_seq)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        # Post-optimization metrics
        metrics_after = self.calculator.compute_metrics(
            optimized.sequence, organism.codon_table
        )

        # Verify amino acid preservation
        optimized_protein = optimized.translate()
        warnings: List[str] = []
        if optimized_protein != protein_seq:
            warnings.append(
                "WARNING: Optimized sequence does not preserve the original "
                "amino acid sequence. This indicates a bug."
            )

        # Check constraints
        constraint_warnings = optimizer.check_constraints(optimized.sequence)
        warnings.extend(constraint_warnings)

        return OptimizationResult(
            original_sequence=clean_seq,
            optimized_dna=optimized,
            protein_sequence=protein_seq,
            organism_name=organism.display_name,
            input_type=input_type,
            metrics_before=metrics_before if metrics_before else {},
            metrics_after=metrics_after,
            warnings=warnings,
        )

    @staticmethod
    def _build_strategy(name: str, seed: int | None = None) -> OptimizationStrategy:
        """Build a strategy instance by name."""
        if name == "highest_frequency":
            return HighestFrequencyStrategy()
        elif name == "weighted_random":
            return WeightedRandomStrategy(seed=seed)
        else:
            raise ValueError(f"Unknown strategy: {name}")
