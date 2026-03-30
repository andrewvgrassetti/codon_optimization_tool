"""Sequence analysis and metrics calculation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from src.config.constants import CODON_TABLE, AMINO_ACID_TO_CODONS
from src.config.organisms import CodonUsageTable
from src.models.sequences import DNASequence


class SequenceAnalyzer:
    """General-purpose sequence analysis utilities."""

    @staticmethod
    def gc_content(dna: str) -> float:
        """Calculate GC content as a fraction."""
        seq = dna.upper()
        if not seq:
            return 0.0
        return sum(1 for b in seq if b in "GC") / len(seq)

    @staticmethod
    def gc_content_by_position(dna: str) -> Dict[str, float]:
        """Calculate GC content at each codon position (1st, 2nd, 3rd).

        Returns a dict with keys 'gc1', 'gc2', 'gc3'.
        """
        seq = dna.upper()
        positions: Dict[str, List[str]] = {"gc1": [], "gc2": [], "gc3": []}
        for i in range(0, len(seq) - 2, 3):
            positions["gc1"].append(seq[i])
            positions["gc2"].append(seq[i + 1])
            positions["gc3"].append(seq[i + 2])

        result: Dict[str, float] = {}
        for key, bases in positions.items():
            if bases:
                result[key] = sum(1 for b in bases if b in "GC") / len(bases)
            else:
                result[key] = 0.0
        return result

    @staticmethod
    def codon_frequency_distribution(dna: str) -> Dict[str, int]:
        """Count occurrences of each codon in a DNA sequence."""
        seq = dna.upper()
        codons = [seq[i : i + 3] for i in range(0, len(seq) - 2, 3)]
        # Exclude stop codons from distribution
        codons = [c for c in codons if CODON_TABLE.get(c, "*") != "*"]
        return dict(Counter(codons))


class CodonMetricsCalculator:
    """Calculate codon-optimization-specific metrics.

    Note: The CAI implementation here is an approximation based on relative
    adaptiveness values derived from the provided codon usage frequencies.
    A fully authoritative CAI would require reference gene sets from the
    target organism, which is beyond the scope of this tool.
    """

    @staticmethod
    def relative_adaptiveness(codon_table: CodonUsageTable) -> Dict[str, float]:
        """Compute relative adaptiveness (w_i) for each codon.

        w_i = frequency(codon) / max_frequency(amino acid)
        """
        w: Dict[str, float] = {}
        for aa, codons in AMINO_ACID_TO_CODONS.items():
            freqs = {c: codon_table.get_frequency(c) for c in codons}
            max_freq = max(freqs.values()) if freqs else 1.0
            for codon, freq in freqs.items():
                if max_freq > 0:
                    w[codon] = freq / max_freq
                else:
                    w[codon] = 1.0 / len(codons)
        return w

    @classmethod
    def cai_score(cls, dna: str, codon_table: CodonUsageTable) -> float:
        """Compute an approximate Codon Adaptation Index (CAI).

        CAI = geometric mean of relative adaptiveness values for all codons.
        Values range from 0 to 1; higher is better adapted.

        This is an approximation: a true CAI uses reference gene sets rather
        than genome-wide codon usage frequencies.
        """
        seq = dna.upper()
        w = cls.relative_adaptiveness(codon_table)

        log_sum = 0.0
        count = 0
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            aa = CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue
            wi = w.get(codon, 0.0)
            if wi > 0:
                log_sum += math.log(wi)
            else:
                # Assign a small value for zero-frequency codons
                log_sum += math.log(0.001)
            count += 1

        if count == 0:
            return 0.0
        return math.exp(log_sum / count)

    @classmethod
    def weighted_rscu(cls, dna: str, codon_table: CodonUsageTable) -> float:
        """Calculate average weighted RSCU for a DNA sequence.

        Weighted RSCU measures codon usage bias relative to a reference
        codon frequency table (the organism's codon usage frequencies).

        For each amino acid present in the sequence, and for each of its
        synonymous codons:
        1. Count occurrences (eff) and sum per amino acid (sum_eff)
        2. Expected frequency = reference_weight * sum_eff
        3. weighted_RSCU = eff / expected_frequency (0 when expected is 0)
        4. Return mean of all per-codon weighted_RSCU values

        Interpretation:
        - Values near **1.0** indicate codon usage closely matches the
          organism's natural codon frequencies (proportional usage).
        - Values **below 1.0** indicate bias toward the most-preferred
          codons. Typical range for highest-frequency optimization is
          ~0.5–0.8 depending on protein amino acid composition (proteins
          rich in highly degenerate amino acids like Leu/Ser/Arg trend
          lower).
        - Values **above 1.0** (up to ~2.0) indicate bias toward less-
          preferred or rare codons.
        """
        seq = dna.upper()

        # Count codons (excluding stop codons)
        codon_counts: Dict[str, int] = defaultdict(int)
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            aa = CODON_TABLE.get(codon)
            if aa is not None and aa != "*":
                codon_counts[codon] += 1

        # Calculate weighted RSCU for each codon grouped by amino acid
        wrscu_values: List[float] = []
        for aa, codons in AMINO_ACID_TO_CODONS.items():
            sum_eff = sum(codon_counts.get(c, 0) for c in codons)
            if sum_eff == 0:
                continue

            for codon in codons:
                eff = codon_counts.get(codon, 0)
                weight = codon_table.get_frequency(codon)
                expected_freq = weight * sum_eff

                if expected_freq > 0:
                    wrscu_values.append(eff / expected_freq)
                else:
                    wrscu_values.append(0.0)

        if not wrscu_values:
            return 0.0
        return sum(wrscu_values) / len(wrscu_values)

    @staticmethod
    def compute_metrics(
        dna: str, codon_table: Optional[CodonUsageTable] = None
    ) -> Dict[str, object]:
        """Compute a comprehensive set of metrics for a DNA sequence.

        Returns a dict with keys like 'length', 'gc_content', 'cai',
        'weighted_rscu', etc.
        """
        analyzer = SequenceAnalyzer()
        metrics: Dict[str, object] = {
            "length_bp": len(dna),
            "length_codons": len(dna) // 3 if dna else 0,
            "gc_content": analyzer.gc_content(dna),
        }

        gc_pos = analyzer.gc_content_by_position(dna)
        metrics.update(gc_pos)

        if codon_table:
            metrics["cai"] = CodonMetricsCalculator.cai_score(dna, codon_table)
            metrics["weighted_rscu"] = CodonMetricsCalculator.weighted_rscu(
                dna, codon_table
            )

        return metrics
