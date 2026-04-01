"""Export utilities for optimization results."""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List

from src.models.sequences import OptimizationResult


class FastaExporter:
    """Export sequences in FASTA format."""

    @staticmethod
    def export(result: OptimizationResult, name: str = "") -> str:
        """Generate FASTA-formatted string for the optimized sequence."""
        seq_name = name or "optimized_sequence"
        header = f">{seq_name} organism={result.organism_name} strategy=codon_optimized"
        seq = result.optimized_dna.sequence
        # Wrap at 80 characters
        lines = [seq[i : i + 80] for i in range(0, len(seq), 80)]
        return header + "\n" + "\n".join(lines) + "\n"


class CsvExporter:
    """Export optimization summary as CSV."""

    @staticmethod
    def export(result: OptimizationResult, name: str = "") -> str:
        """Generate CSV string with optimization metrics."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Sequence Name", name or "optimized_sequence"])
        writer.writerow(["Organism", result.organism_name])
        writer.writerow(["Input Type", result.input_type])
        writer.writerow(["Protein Length", len(result.protein_sequence)])
        writer.writerow(["Optimized DNA Length (bp)", len(result.optimized_dna)])

        # Metrics after optimization
        if result.metrics_after:
            for key, value in result.metrics_after.items():
                if isinstance(value, float):
                    writer.writerow([f"Optimized {key}", f"{value:.4f}"])
                else:
                    writer.writerow([f"Optimized {key}", value])

        # Metrics before (if DNA input)
        if result.metrics_before:
            for key, value in result.metrics_before.items():
                if isinstance(value, float):
                    writer.writerow([f"Original {key}", f"{value:.4f}"])
                else:
                    writer.writerow([f"Original {key}", value])

        if result.warnings:
            writer.writerow(["Warnings", "; ".join(result.warnings)])

        return output.getvalue()


class TextExporter:
    """Export results as plain text."""

    @staticmethod
    def export(result: OptimizationResult, name: str = "") -> str:
        """Generate a plain text summary."""
        lines = [
            f"Codon Optimization Report",
            f"========================",
            f"Sequence: {name or 'optimized_sequence'}",
            f"Target Organism: {result.organism_name}",
            f"Input Type: {result.input_type}",
            f"",
            f"Protein Sequence ({len(result.protein_sequence)} aa):",
            result.protein_sequence,
            f"",
            f"Optimized DNA Sequence ({len(result.optimized_dna)} bp):",
            result.optimized_dna.sequence,
            f"",
        ]

        if result.metrics_after:
            lines.append("Optimization Metrics:")
            for key, value in result.metrics_after.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        if result.metrics_before:
            lines.append("Original Metrics:")
            for key, value in result.metrics_before.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        if result.warnings:
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        return "\n".join(lines)


class MultiVariantCsvExporter:
    """Export multiple optimization results as a single CSV with one row per variant."""

    @staticmethod
    def export(
        results: List[tuple[str, OptimizationResult | None, str]],
    ) -> str:
        """Generate a CSV string with one row per variant.

        Columns match the user-requested layout: name, optimization strategy,
        %GC range, %GC of CDS, %GC 1, %GC 2, %GC 3, CAI, wRSCU, sequence.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "name",
            "optimization strategy",
            "%GC range",
            "%GC of CDS",
            "%GC 1",
            "%GC 2",
            "%GC 3",
            "CAI",
            "wRSCU",
            "sequence",
        ])

        for name, result, error in results:
            if error or result is None:
                continue
            metrics = result.metrics_after or {}
            gc_range = ""
            if result.variant_label:
                gc_match = re.search(r"GC (\d+%–\d+%)", result.variant_label)
                if gc_match:
                    gc_range = gc_match.group(1)

            strategy_map = {
                "highest_frequency": "Highest Frequency",
                "weighted_random": "Weighted Random",
            }
            strategy = strategy_map.get(result.strategy_name, result.strategy_name)

            gc_content = metrics.get("gc_content")
            gc1 = metrics.get("gc1")
            gc2 = metrics.get("gc2")
            gc3 = metrics.get("gc3")
            cai = metrics.get("cai")
            wrscu = metrics.get("weighted_rscu")

            writer.writerow([
                name,
                strategy,
                gc_range,
                f"{gc_content:.4f}" if isinstance(gc_content, float) else "",
                f"{gc1:.4f}" if isinstance(gc1, float) else "",
                f"{gc2:.4f}" if isinstance(gc2, float) else "",
                f"{gc3:.4f}" if isinstance(gc3, float) else "",
                f"{cai:.4f}" if isinstance(cai, float) else "",
                f"{wrscu:.4f}" if isinstance(wrscu, float) else "",
                result.optimized_dna.sequence,
            ])

        return output.getvalue()
