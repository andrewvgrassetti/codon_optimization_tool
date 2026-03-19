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
