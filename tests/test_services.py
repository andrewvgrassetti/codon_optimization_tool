"""Tests for the high-level optimization service."""

import pytest

from src.services.optimization_service import OptimizationService


@pytest.fixture
def service():
    return OptimizationService()


class TestOptimizationService:
    def test_get_organisms(self, service):
        organisms = service.get_organisms()
        assert len(organisms) >= 4
        names = [o.name for o in organisms]
        assert "e_coli" in names
        assert "human" in names

    def test_validate_dna_valid(self, service):
        result = service.validate_sequence("ATGAAAGCCTAA", "dna")
        assert result.is_valid

    def test_validate_protein_valid(self, service):
        result = service.validate_sequence("MKFLV", "protein")
        assert result.is_valid

    def test_optimize_protein(self, service):
        result = service.optimize(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
        )
        assert result.optimized_dna is not None
        assert result.protein_sequence == "MKFLV"
        assert result.optimized_dna.translate() == "MKFLV"
        assert result.metrics_after

    def test_optimize_dna(self, service):
        result = service.optimize(
            sequence="ATGAAATTTCTGGTGTAA",
            input_type="dna",
            organism_name="e_coli",
        )
        assert result.optimized_dna is not None
        assert result.metrics_before
        assert result.metrics_after

    def test_optimize_preserves_protein_from_dna(self, service):
        from src.models.sequences import DNASequence

        dna = "ATGAAATTTCTGGTGTAA"
        original_protein = DNASequence(sequence=dna).translate()
        result = service.optimize(
            sequence=dna,
            input_type="dna",
            organism_name="human",
        )
        assert result.optimized_dna.translate() == original_protein

    def test_optimize_unknown_organism(self, service):
        with pytest.raises(ValueError, match="Unknown organism"):
            service.optimize(
                sequence="MKFLV",
                input_type="protein",
                organism_name="martian",
            )

    def test_optimize_with_constraints(self, service):
        from src.optimization.constraints import GCContentConstraint

        result = service.optimize(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            constraints=[GCContentConstraint(min_gc=0.90, max_gc=1.0)],
        )
        # The constraint should produce a warning since GC can't be 90%
        assert any("GC content" in w for w in result.warnings)

    def test_weighted_random_strategy(self, service):
        result = service.optimize(
            sequence="MKFLVDTY",
            input_type="protein",
            organism_name="e_coli",
            strategy_name="weighted_random",
            seed=42,
        )
        assert result.optimized_dna.translate() == "MKFLVDTY"
