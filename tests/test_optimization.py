"""Tests for codon optimization logic."""

import pytest

from src.config.organisms import get_default_registry
from src.models.sequences import DNASequence
from src.optimization.constraints import (
    GCContentConstraint,
    HomopolymerConstraint,
    MotifConstraint,
    RestrictionSiteConstraint,
    WRSCUConstraint,
)
from src.optimization.optimizer import CodonOptimizer
from src.optimization.strategies import HighestFrequencyStrategy, WeightedRandomStrategy, RandomOptimizationStrategy


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

    def test_wrscu_too_low(self, ecoli_profile):
        c = WRSCUConstraint(
            codon_table=ecoli_profile.codon_table,
            min_wrscu=0.90,
            max_wrscu=1.50,
        )
        # Highest-frequency codons produce wRSCU well below 0.90
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        result = optimizer.optimize_from_protein("MKFLVDTY")
        warnings = c.check(result.sequence)
        assert len(warnings) == 1
        assert "below" in warnings[0]

    def test_wrscu_within_range(self, ecoli_profile):
        c = WRSCUConstraint(
            codon_table=ecoli_profile.codon_table,
            min_wrscu=0.10,
            max_wrscu=2.00,
        )
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        result = optimizer.optimize_from_protein("MKFLVDTY")
        warnings = c.check(result.sequence)
        assert len(warnings) == 0

    def test_wrscu_empty_sequence(self, ecoli_profile):
        c = WRSCUConstraint(
            codon_table=ecoli_profile.codon_table,
            min_wrscu=0.50,
            max_wrscu=1.50,
        )
        warnings = c.check("")
        assert len(warnings) == 0

    def test_wrscu_name(self, ecoli_profile):
        c = WRSCUConstraint(
            codon_table=ecoli_profile.codon_table,
            min_wrscu=0.50,
            max_wrscu=1.50,
        )
        assert "wRSCU" in c.name()
        assert "0.50" in c.name()
        assert "1.50" in c.name()

    def test_no_stop_codon_by_default(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        result = optimizer.optimize_from_protein("MK")
        # Should NOT end with a stop codon
        assert not result.sequence.endswith("TAA")
        assert not result.sequence.endswith("TAG")
        assert not result.sequence.endswith("TGA")
        # Should still encode the protein
        assert result.translate() == "MK"

    def test_stop_codon_opt_in(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
            add_stop_codon=True,
        )
        result = optimizer.optimize_from_protein("MK")
        assert result.sequence.endswith("TAA")

    def test_optimize_from_dna_no_stop_codon(self, ecoli_profile):
        optimizer = CodonOptimizer(
            organism=ecoli_profile,
            strategy=HighestFrequencyStrategy(),
        )
        # Input DNA has stop codon, output should NOT
        dna = "ATGAAATAA"
        result = optimizer.optimize_from_dna(dna)
        assert result.translate() == "MK"
        assert not result.sequence.endswith("TAA")
        assert not result.sequence.endswith("TAG")
        assert not result.sequence.endswith("TGA")


class TestWeightedRandomWithConstraints:
    """WeightedRandomStrategy rejection-sampling behaviour."""

    def test_without_constraints_returns_none_from_full_sequence(self, ecoli_profile):
        """No constraints → optimize_full_sequence returns None (per-codon path)."""
        strategy = WeightedRandomStrategy(seed=7)
        result = strategy.optimize_full_sequence("MKFLV", ecoli_profile.codon_table)
        assert result is None

    def test_with_gc_constraints_produces_sequence_in_range(self, human_profile):
        """With an achievable GC range the strategy should find a valid sequence."""
        # Use a wide-but-reachable GC range for a moderate protein
        strategy = WeightedRandomStrategy(
            gc_min=0.40, gc_max=0.65, max_attempts=500
        )
        optimizer = CodonOptimizer(organism=human_profile, strategy=strategy)
        protein = "MKFLVDTYWSCRHPQEINA"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein
        gc = result.gc_content
        assert 0.40 <= gc <= 0.65, f"GC {gc:.3f} outside [0.40, 0.65]"

    def test_with_wrscu_constraints_produces_sequence_in_range(self, human_profile):
        """With an achievable wRSCU range the strategy should find a valid sequence."""
        from src.analysis.metrics import CodonMetricsCalculator
        # wRSCU ~0.80–1.10 is achievable via weighted-random for human
        strategy = WeightedRandomStrategy(
            wrscu_min=0.80, wrscu_max=1.10, max_attempts=500
        )
        optimizer = CodonOptimizer(organism=human_profile, strategy=strategy)
        protein = "MKFLVDTYWSCRHPQEINA"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein
        wrscu = CodonMetricsCalculator.weighted_rscu(
            result.sequence, human_profile.codon_table
        )
        assert 0.80 <= wrscu <= 1.10, f"wRSCU {wrscu:.3f} outside [0.80, 1.10]"

    def test_fallback_best_candidate_with_warning_when_impossible(self, ecoli_profile):
        """When constraints cannot be satisfied the best candidate + warning are returned."""
        # GC > 99% is effectively impossible for a real protein
        strategy = WeightedRandomStrategy(
            gc_min=0.99, gc_max=1.0, max_attempts=20
        )
        optimizer = CodonOptimizer(organism=ecoli_profile, strategy=strategy)
        result = optimizer.optimize_from_protein("MKFLV")
        # Must still encode the correct protein
        assert result.translate() == "MKFLV"
        # Strategy should have recorded a fallback warning
        assert len(strategy.last_warnings) > 0
        assert "could not satisfy" in strategy.last_warnings[0].lower()

    def test_seeded_reproducibility_without_constraints(self, ecoli_profile):
        """Without constraints, seeded behaviour is unchanged."""
        s1 = WeightedRandomStrategy(seed=99)
        s2 = WeightedRandomStrategy(seed=99)
        opt1 = CodonOptimizer(organism=ecoli_profile, strategy=s1)
        opt2 = CodonOptimizer(organism=ecoli_profile, strategy=s2)
        r1 = opt1.optimize_from_protein("MKFLVDTY")
        r2 = opt2.optimize_from_protein("MKFLVDTY")
        assert r1.sequence == r2.sequence

    def test_preserves_amino_acids_with_constraints(self, human_profile):
        """Rejection-sampling must preserve the exact amino acid sequence."""
        strategy = WeightedRandomStrategy(gc_min=0.45, gc_max=0.60)
        optimizer = CodonOptimizer(organism=human_profile, strategy=strategy)
        protein = "MKFLVDTY"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein


class TestRandomOptimizationStrategy:
    """RandomOptimizationStrategy behaviour."""

    def test_uniform_selection_without_constraints(self, ecoli_profile):
        """Without constraints uses per-codon uniform random (optimize_full_sequence → None)."""
        strategy = RandomOptimizationStrategy(seed=0)
        assert strategy.optimize_full_sequence("MKFLV", ecoli_profile.codon_table) is None

    def test_preserves_amino_acids_without_constraints(self, ecoli_profile):
        strategy = RandomOptimizationStrategy(seed=42)
        optimizer = CodonOptimizer(organism=ecoli_profile, strategy=strategy)
        protein = "MKFLVDTYWSCRHP"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein

    def test_with_gc_constraints_produces_sequence_in_range(self, human_profile):
        """With an achievable GC range the strategy should find a valid sequence."""
        strategy = RandomOptimizationStrategy(
            gc_min=0.35, gc_max=0.65, max_attempts=500
        )
        optimizer = CodonOptimizer(organism=human_profile, strategy=strategy)
        protein = "MKFLVDTYWSCRHPQEINA"
        result = optimizer.optimize_from_protein(protein)
        assert result.translate() == protein
        gc = result.gc_content
        assert 0.35 <= gc <= 0.65, f"GC {gc:.3f} outside [0.35, 0.65]"

    def test_fallback_best_candidate_with_warning_when_impossible(self, ecoli_profile):
        """When constraints cannot be satisfied the best candidate + warning are returned."""
        strategy = RandomOptimizationStrategy(
            gc_min=0.99, gc_max=1.0, max_attempts=20
        )
        optimizer = CodonOptimizer(organism=ecoli_profile, strategy=strategy)
        result = optimizer.optimize_from_protein("MKFLV")
        assert result.translate() == "MKFLV"
        assert len(strategy.last_warnings) > 0
        assert "could not satisfy" in strategy.last_warnings[0].lower()

    def test_differs_from_weighted_random(self, ecoli_profile):
        """Uniform selection should differ from weighted-random on average."""
        protein = "MKFLVDTYWSCRHPQEINALMVG" * 3
        seeded_weighted = []
        seeded_uniform = []
        for seed in range(10):
            wopt = CodonOptimizer(
                organism=ecoli_profile,
                strategy=WeightedRandomStrategy(seed=seed),
            )
            ropt = CodonOptimizer(
                organism=ecoli_profile,
                strategy=RandomOptimizationStrategy(seed=seed),
            )
            seeded_weighted.append(wopt.optimize_from_protein(protein).sequence)
            seeded_uniform.append(ropt.optimize_from_protein(protein).sequence)
        # At least some sequences should differ between the two strategies
        assert seeded_weighted != seeded_uniform

    def test_seeded_reproducibility(self, ecoli_profile):
        """Same seed produces the same sequence."""
        s1 = RandomOptimizationStrategy(seed=7)
        s2 = RandomOptimizationStrategy(seed=7)
        opt1 = CodonOptimizer(organism=ecoli_profile, strategy=s1)
        opt2 = CodonOptimizer(organism=ecoli_profile, strategy=s2)
        r1 = opt1.optimize_from_protein("MKFLVDTY")
        r2 = opt2.optimize_from_protein("MKFLVDTY")
        assert r1.sequence == r2.sequence


class TestVariantConfigLabelRandomOptimization:
    """VariantConfig.label for the new strategy."""

    def test_label_random_optimization(self):
        from src.models.sequences import VariantConfig
        config = VariantConfig(strategy_name="random_optimization")
        assert config.label == "Random Optimization"

    def test_label_random_optimization_with_gc_range(self):
        from src.models.sequences import VariantConfig
        config = VariantConfig(
            strategy_name="random_optimization", gc_min=0.40, gc_max=0.60
        )
        assert "Random Optimization" in config.label
        assert "GC 40%–60%" in config.label

    def test_label_random_optimization_with_wrscu_range(self):
        from src.models.sequences import VariantConfig
        config = VariantConfig(
            strategy_name="random_optimization", wrscu_min=0.50, wrscu_max=1.50
        )
        assert "Random Optimization" in config.label
        assert "wRSCU 0.50–1.50" in config.label


class TestServiceWithNewStrategies:
    """Service-level integration tests for the new/updated strategies."""

    def test_weighted_random_with_gc_constraint_via_service(self, human_profile):
        """weighted_random + GC constraints uses rejection sampling in the service."""
        from src.services.optimization_service import OptimizationService
        from src.models.sequences import VariantConfig
        service = OptimizationService()
        configs = [
            VariantConfig(
                strategy_name="weighted_random",
                gc_min=0.40,
                gc_max=0.65,
            )
        ]
        results = service.optimize_variants(
            sequence="MKFLVDTYWSCRHPQEINA",
            input_type="protein",
            organism_name="human",
            variant_configs=configs,
        )
        assert len(results) == 1
        gc = results[0].optimized_dna.gc_content
        assert 0.40 <= gc <= 0.65, f"GC {gc:.3f} outside expected range"

    def test_random_optimization_strategy_via_service(self, ecoli_profile):
        """random_optimization strategy is reachable through the service."""
        from src.services.optimization_service import OptimizationService
        from src.models.sequences import VariantConfig
        service = OptimizationService()
        configs = [VariantConfig(strategy_name="random_optimization")]
        results = service.optimize_variants(
            sequence="MKFLVDTY",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 1
        assert results[0].optimized_dna.translate() == "MKFLVDTY"
        assert "Random Optimization" in results[0].variant_label

    def test_random_optimization_with_gc_constraint_via_service(self, human_profile):
        """random_optimization + GC constraints uses rejection sampling in the service."""
        from src.services.optimization_service import OptimizationService
        from src.models.sequences import VariantConfig
        service = OptimizationService()
        configs = [
            VariantConfig(
                strategy_name="random_optimization",
                gc_min=0.35,
                gc_max=0.65,
            )
        ]
        results = service.optimize_variants(
            sequence="MKFLVDTYWSCRHPQEINA",
            input_type="protein",
            organism_name="human",
            variant_configs=configs,
        )
        assert len(results) == 1
        gc = results[0].optimized_dna.gc_content
        assert 0.35 <= gc <= 0.65, f"GC {gc:.3f} outside expected range"

