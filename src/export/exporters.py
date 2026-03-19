"""Export utilities for optimization results."""

from __future__ import annotations

import csv
import io
from typing import Dict

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
