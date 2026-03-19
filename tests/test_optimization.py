"""Tests for codon optimization logic."""

import pytest

from src.config.organisms import get_default_registry
from src.models.sequences import DNASequence
from src.optimization.constraints import (
    GCContentConstraint,
    HomopolymerConstraint,
    MotifConstraint,
    RestrictionSiteConstraint,
)
from src.optimization.optimizer import CodonOptimizer
from src.optimization.strategies import HighestFrequencyStrategy, WeightedRandomStrategy


@pytest.fixture
def ecoli_profile():
    registry = get_default_registry()
    return registry.get("e_coli")


@pytest.fixture
def human_profile():
    registry = get_default_registry()
    return registry.get("human")


class TestHighestFrequencyStrategy:
    def test_preserves_amino_acids_from_protein(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        protein = "MKFLVDTY"
        result = optimizer.optimize_from_protein(protein)
        translated = result.translate()
        assert translated == protein

    def test_preserves_amino_acids_from_dna(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        dna = "ATGAAATTTCTGGTGGATACCTATTAA"
        source = DNASequence(sequence=dna)
        original_protein = source.translate()
        result = optimizer.optimize_from_dna(dna)
        optimized_protein = result.translate()
        assert optimized_protein == original_protein

    def test_deterministic(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        r1 = optimizer.optimize_from_protein("MKFLVDTY")
        r2 = optimizer.optimize_from_protein("MKFLVDTY")
        assert r1.sequence == r2.sequence

    def test_uses_preferred_codons(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        # Leucine: CTG is most frequent in E. coli
        result = optimizer.optimize_from_protein("L")
        codons = result.get_codons()
        assert codons[0] == "CTG"


class TestWeightedRandomStrategy:
    def test_preserves_amino_acids(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=WeightedRandomStrategy(seed=42),
        )
        protein = "MKFLVDTYWSCRHP"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein

    def test_seeded_reproducibility(self, ecoli_profile):
        opt1 = CodonOptimizer(
            organism=ecoli_profile,
            strategy=WeightedRandomStrategy(seed=99),
        )
        opt2 = CodonOptimizer(
            organism=ecoli_profile,
            strategy=WeightedRandomStrategy(seed=99),
        )
        r1 = opt1.optimize_from_protein("MKFLVDTY")
        r2 = opt2.optimize_from_protein("MKFLVDTY")
        assert r1.sequence == r2.sequence


class TestOptimizationWithDifferentOrganisms:
    def test_ecoli_vs_human(self, ecoli_profile, human_profile):
        protein = "MKFLV"
        opt_ecoli = CodonOptimizer(
            organism=ecoli_profile, strategy=HighestFrequencyStrategy()
        )
        opt_human = CodonOptimizer(
            organism=human_profile, strategy=HighestFrequencyStrategy()
        )
        r_ecoli = opt_ecoli.optimize_from_protein(protein)
        r_human = opt_human.optimize_from_protein(protein)
        # Both must preserve the protein
        assert r_ecoli.translate() == protein
        assert r_human.translate() == protein
        # They should differ since organisms use different codons
        assert r_ecoli.sequence != r_human.sequence


class TestConstraints:
    def test_gc_content_too_low(self):
        c = GCContentConstraint(min_gc=0.50, max_gc=0.70)
        # All A and T → GC = 0%
        warnings = c.check("AAAAATTTTT")
        assert len(warnings) == 1
        assert "below" in warnings[0]

    def test_gc_content_too_high(self):
        c = GCContentConstraint(min_gc=0.30, max_gc=0.50)
        warnings = c.check("GCGCGCGCGC")
        assert len(warnings) == 1
        assert "above" in warnings[0]

    def test_gc_content_ok(self):
        c = GCContentConstraint(min_gc=0.30, max_gc=0.70)
        warnings = c.check("ATGCATGC")
        assert len(warnings) == 0

    def test_restriction_site_detected(self):
        c = RestrictionSiteConstraint(sites_to_avoid={"EcoRI": "GAATTC"})
        warnings = c.check("ATGGAATTCATG")
        assert len(warnings) == 1
        assert "EcoRI" in warnings[0]

    def test_restriction_site_absent(self):
        c = RestrictionSiteConstraint(sites_to_avoid={"EcoRI": "GAATTC"})
        warnings = c.check("ATGAAATTTGCC")
        assert len(warnings) == 0

    def test_homopolymer_detected(self):
        c = HomopolymerConstraint(max_run_length=5)
        warnings = c.check("ATGAAAAAAGCC")
        assert len(warnings) == 1
        assert "Homopolymer" in warnings[0]

    def test_homopolymer_ok(self):
        c = HomopolymerConstraint(max_run_length=5)
        warnings = c.check("ATGAAAGCC")
        assert len(warnings) == 0

    def test_motif_detected(self):
        c = MotifConstraint(forbidden_motifs=["AATAAA"])
        warnings = c.check("ATGAATAAAGCC")
        assert len(warnings) == 1

    def test_motif_absent(self):
        c = MotifConstraint(forbidden_motifs=["AATAAA"])
        warnings = c.check("ATGAAAGCC")
        assert len(warnings) == 0
