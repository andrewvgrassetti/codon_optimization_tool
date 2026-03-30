"""Reusable Streamlit UI components."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.metrics import SequenceAnalyzer
from src.config.constants import AMINO_ACID_ABBREVIATIONS, CODON_TABLE
from src.models.sequences import OptimizationResult


def render_metrics_cards(
    metrics: Dict[str, object], title: str = "Metrics"
) -> None:
    """Render a row of metric cards."""
    st.subheader(title)
    if not metrics:
        st.info("No metrics available.")
        return

    display_keys = {
        "length_bp": ("Length (bp)", None),
        "length_codons": ("Codons", None),
        "gc_content": ("GC Content", ".1%"),
        "gc1": ("GC1", ".1%"),
        "gc2": ("GC2", ".1%"),
        "gc3": ("GC3", ".1%"),
        "cai": ("CAI Score", ".3f"),
    }

    cols = st.columns(min(len(display_keys), 4))
    col_idx = 0
    for key, (label, fmt) in display_keys.items():
        if key in metrics:
            val = metrics[key]
            if fmt and isinstance(val, (int, float)):
                display_val = f"{val:{fmt}}"
            else:
                display_val = str(val)
            cols[col_idx % len(cols)].metric(label, display_val)
            col_idx += 1


def render_comparison_metrics(
    before: Dict[str, object], after: Dict[str, object]
) -> None:
    """Render side-by-side metric comparison for before/after optimization."""
    st.subheader("📊 Metrics Comparison")

    rows = []
    keys = ["gc_content", "gc1", "gc2", "gc3", "cai"]
    labels = {
        "gc_content": "GC Content",
        "gc1": "GC Position 1",
        "gc2": "GC Position 2",
        "gc3": "GC Position 3",
        "cai": "CAI Score",
    }

    for key in keys:
        before_val = before.get(key)
        after_val = after.get(key)
        if before_val is not None or after_val is not None:
            fmt = ".1%" if key.startswith("gc") else ".3f"
            rows.append({
                "Metric": labels.get(key, key),
                "Original": f"{before_val:{fmt}}" if before_val is not None else "N/A",
                "Optimized": f"{after_val:{fmt}}" if after_val is not None else "N/A",
            })

    if rows:
        df = pd.DataFrame(rows)
        st.table(df)


def _codon_label(codon: str) -> str:
    """Return a codon label with its amino acid abbreviation, e.g. 'ATG (Met)'."""
    aa = CODON_TABLE.get(codon.upper(), "")
    abbrev = AMINO_ACID_ABBREVIATIONS.get(aa, "")
    if abbrev:
        return f"{codon} ({abbrev})"
    return codon


def render_codon_usage_chart(
    dna_sequence: str, title: str = "Codon Usage Distribution"
) -> None:
    """Render a bar chart of codon usage frequencies."""
    dist = SequenceAnalyzer.codon_frequency_distribution(dna_sequence)
    if not dist:
        return

    sorted_codons = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    codons = [_codon_label(c) for c, _ in sorted_codons[:30]]  # Top 30 for readability
    counts = [n for _, n in sorted_codons[:30]]

    fig = go.Figure(data=[
        go.Bar(x=codons, y=counts, marker_color="steelblue")
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Codon",
        yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sequence_display(
    sequence: str, label: str = "Sequence", wrap: int = 80
) -> None:
    """Display a sequence in a formatted text area."""
    wrapped = "\n".join(
        sequence[i : i + wrap] for i in range(0, len(sequence), wrap)
    )
    st.text_area(label, wrapped, height=150, disabled=True)


def render_warnings(warnings: list[str]) -> None:
    """Display warning messages."""
    if warnings:
        for w in warnings:
            st.warning(w)
