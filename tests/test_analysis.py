"""Tests for sequence analysis and metrics."""

import pytest

from src.analysis.metrics import CodonMetricsCalculator, SequenceAnalyzer
from src.config.organisms import get_default_registry


@pytest.fixture
def ecoli_table():
    registry = get_default_registry()
    return registry.get("e_coli").codon_table


class TestSequenceAnalyzer:
    def test_gc_content_all_gc(self):
        assert SequenceAnalyzer.gc_content("GCGCGC") == 1.0

    def test_gc_content_no_gc(self):
        assert SequenceAnalyzer.gc_content("ATATAT") == 0.0

    def test_gc_content_mixed(self):
        assert abs(SequenceAnalyzer.gc_content("ATGC") - 0.5) < 0.001

    def test_gc_content_empty(self):
        assert SequenceAnalyzer.gc_content("") == 0.0

    def test_gc_by_position(self):
        # ATG GCC → pos1: A,G; pos2: T,C; pos3: G,C
        result = SequenceAnalyzer.gc_content_by_position("ATGGCC")
        assert result["gc1"] == 0.5  # G out of A,G
        assert result["gc2"] == 0.5  # C out of T,C
        assert result["gc3"] == 1.0  # G,C out of G,C

    def test_codon_frequency_distribution(self):
        dist = SequenceAnalyzer.codon_frequency_distribution("ATGATGATG")
        assert dist.get("ATG", 0) == 3


class TestCodonMetricsCalculator:
    def test_cai_perfect(self, ecoli_table):
        """Sequence built from the best codon for each aa should have CAI ~1.0."""
        # Build an optimal sequence for a short protein
        from src.optimization.optimizer import CodonOptimizer
        from src.optimization.strategies import HighestFrequencyStrategy
        from src.config.organisms import get_default_registry

        ecoli = get_default_registry().get("e_coli")
        optimizer = CodonOptimizer(
            organism=ecoli, strategy=HighestFrequencyStrategy()
        )
        result = optimizer.optimize_from_protein("MKFLVDTY")
        cai = CodonMetricsCalculator.cai_score(result.sequence, ecoli_table)
        assert cai > 0.95  # Should be very close to 1.0

    def test_cai_range(self, ecoli_table):
        """CAI should be between 0 and 1."""
        cai = CodonMetricsCalculator.cai_score("ATGAAATTT", ecoli_table)
        assert 0.0 < cai <= 1.0

    def test_compute_metrics(self, ecoli_table):
        metrics = CodonMetricsCalculator.compute_metrics("ATGAAAGCC", ecoli_table)
        assert metrics["length_bp"] == 9
        assert metrics["length_codons"] == 3
        assert "gc_content" in metrics
        assert "cai" in metrics
        assert "weighted_rscu" in metrics


class TestWeightedRSCU:
    def test_wrscu_positive(self, ecoli_table):
        """Weighted RSCU should be a positive value for a valid sequence."""
        wrscu = CodonMetricsCalculator.weighted_rscu("ATGAAAGCC", ecoli_table)
        assert wrscu > 0.0

    def test_wrscu_empty_sequence(self, ecoli_table):
        """Empty sequence should return 0.0."""
        wrscu = CodonMetricsCalculator.weighted_rscu("", ecoli_table)
        assert wrscu == 0.0

    def test_wrscu_optimal_below_one(self, ecoli_table):
        """Highest-frequency codons should produce wRSCU below 1.0."""
        from src.optimization.optimizer import CodonOptimizer
        from src.optimization.strategies import HighestFrequencyStrategy
        from src.config.organisms import get_default_registry

        ecoli = get_default_registry().get("e_coli")
        optimizer = CodonOptimizer(
            organism=ecoli, strategy=HighestFrequencyStrategy()
        )
        result = optimizer.optimize_from_protein("MKFLVDTYARSNQEHGPWIC")
        wrscu = CodonMetricsCalculator.weighted_rscu(
            result.sequence, ecoli_table
        )
        assert 0.4 < wrscu < 1.0

    def test_wrscu_proportional_near_one(self, ecoli_table):
        """Codons selected proportionally to usage should give wRSCU near 1.0."""
        from src.optimization.optimizer import CodonOptimizer
        from src.optimization.strategies import WeightedRandomStrategy
        from src.config.organisms import get_default_registry

        ecoli = get_default_registry().get("e_coli")
        optimizer = CodonOptimizer(
            organism=ecoli, strategy=WeightedRandomStrategy(seed=42)
        )
        # Use a long protein for statistical convergence
        long_protein = "MKFLVDTYARSNQEHGPWIC" * 100
        result = optimizer.optimize_from_protein(long_protein)
        wrscu = CodonMetricsCalculator.weighted_rscu(
            result.sequence, ecoli_table
        )
        assert 0.9 < wrscu < 1.1

    def test_wrscu_suboptimal_above_one(self, ecoli_table):
        """Least-preferred codons should produce wRSCU above 1.0 (up to ~2)."""
        from src.config.constants import AMINO_ACID_TO_CODONS

        # Build a sequence using the worst codon for each amino acid
        protein = "MKFLVDTYARSNQEHGPWIC"
        dna = []
        for aa in protein:
            codons = AMINO_ACID_TO_CODONS.get(aa, [])
            worst = min(codons, key=lambda c: ecoli_table.get_frequency(c))
            dna.append(worst)
        dna_seq = "".join(dna)

        wrscu = CodonMetricsCalculator.weighted_rscu(dna_seq, ecoli_table)
        assert 1.0 < wrscu <= 2.5

    def test_wrscu_in_compute_metrics(self, ecoli_table):
        """compute_metrics should include weighted_rscu when codon table is given."""
        metrics = CodonMetricsCalculator.compute_metrics("ATGAAAGCC", ecoli_table)
        assert "weighted_rscu" in metrics
        assert isinstance(metrics["weighted_rscu"], float)
        assert metrics["weighted_rscu"] > 0.0

    def test_wrscu_no_codon_table(self):
        """compute_metrics without codon table should not include weighted_rscu."""
        metrics = CodonMetricsCalculator.compute_metrics("ATGAAAGCC")
        assert "weighted_rscu" not in metrics
