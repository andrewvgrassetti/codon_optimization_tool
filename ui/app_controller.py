"""Streamlit application controller – main orchestration layer for the UI."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from src.config.constants import COMMON_RESTRICTION_SITES, VALID_DNA_BASES, VALID_PROTEIN_CHARS
from src.export.exporters import CsvExporter, FastaExporter, TextExporter
from src.models.sequences import OptimizationResult
from src.optimization.constraints import (
    GCContentConstraint,
    HomopolymerConstraint,
    MotifConstraint,
    OptimizationConstraint,
    RestrictionSiteConstraint,
)
from src.services.optimization_service import OptimizationService
from src.validation.parsers import FastaParser
from ui.components import (
    render_codon_usage_chart,
    render_comparison_metrics,
    render_metrics_cards,
    render_sequence_display,
    render_warnings,
)


class StreamlitApp:
    """Main Streamlit application controller."""

    def __init__(self) -> None:
        self.service = OptimizationService()

    def run(self) -> None:
        """Main entry point for the Streamlit app."""
        st.set_page_config(
            page_title="Codon Optimization Tool",
            page_icon="🧬",
            layout="wide",
        )

        st.title("🧬 Codon Optimization Tool")
        st.markdown(
            "Optimize codon usage for your DNA or protein sequences "
            "targeting a specific expression host organism."
        )

        # Sidebar configuration
        organism, strategy, constraints = self._render_sidebar()

        # Main workspace
        self._render_main_workspace(organism, strategy, constraints)

    def _render_sidebar(self):
        """Render the sidebar configuration panel."""
        st.sidebar.header("⚙️ Configuration")

        # Organism selection – default to Human
        organisms = self.service.get_organisms()
        organism_options = {org.display_name: org.name for org in organisms}
        display_names = list(organism_options.keys())
        human_index = next(
            (i for i, name in enumerate(display_names) if "human" in name.lower()), 0
        )
        selected_display = st.sidebar.selectbox(
            "Target Organism",
            options=display_names,
            index=human_index,
            help="Choose the expression host for codon optimization.",
        )
        organism_name = organism_options[selected_display]

        # Strategy
        strategy = st.sidebar.selectbox(
            "Optimization Strategy",
            options=["Highest Frequency", "Weighted Random"],
            help=(
                "**Highest Frequency**: Always picks the most-used codon (deterministic).\n\n"
                "**Weighted Random**: Picks codons weighted by usage frequency (stochastic)."
            ),
        )
        strategy_key = (
            "highest_frequency" if strategy == "Highest Frequency" else "weighted_random"
        )

        # Constraints
        st.sidebar.header("🔧 Constraints")

        constraints: List[OptimizationConstraint] = []

        if st.sidebar.checkbox("GC Content Range"):
            gc_min = st.sidebar.slider("Min GC%", 0.0, 1.0, 0.30, 0.05)
            gc_max = st.sidebar.slider("Max GC%", 0.0, 1.0, 0.70, 0.05)
            constraints.append(GCContentConstraint(min_gc=gc_min, max_gc=gc_max))

        if st.sidebar.checkbox("Avoid Restriction Sites", value=True):
            available_sites = sorted(COMMON_RESTRICTION_SITES.keys())
            selected_sites = st.sidebar.multiselect(
                "Restriction Enzymes",
                options=available_sites,
                default=["EcoRI", "BamHI", "HindIII", "BspQI"],
                format_func=lambda name: f"{name} ({COMMON_RESTRICTION_SITES[name]})",
            )
            sites_dict = {
                name: COMMON_RESTRICTION_SITES[name] for name in selected_sites
            }
            constraints.append(RestrictionSiteConstraint(sites_to_avoid=sites_dict))

        if st.sidebar.checkbox("Avoid Homopolymer Runs"):
            max_run = st.sidebar.number_input(
                "Max homopolymer length", min_value=3, max_value=15, value=6
            )
            constraints.append(HomopolymerConstraint(max_run_length=max_run))

        if st.sidebar.checkbox("Avoid Custom Motifs"):
            motif_text = st.sidebar.text_input(
                "Forbidden motifs (comma-separated)",
                placeholder="e.g., AATAAA, GGATCC",
            )
            if motif_text.strip():
                motifs = [m.strip() for m in motif_text.split(",") if m.strip()]
                constraints.append(MotifConstraint(forbidden_motifs=motifs))

        return organism_name, strategy_key, constraints

    @staticmethod
    def _detect_sequence_type(sequence: str) -> str:
        """Auto-detect whether a sequence is DNA or protein.

        Heuristic: if the sequence consists only of A, T, C, G (and length is
        divisible by 3), treat it as DNA; otherwise treat it as protein.
        """
        seq = sequence.upper().strip()
        if not seq:
            return "protein"
        non_dna = set(seq) - set(VALID_DNA_BASES)
        if not non_dna and len(seq) % 3 == 0:
            return "dna"
        # If only contains ATCG but length not divisible by 3, still treat as
        # DNA and let the validator surface an error.
        if not non_dna:
            return "dna"
        return "protein"

    def _render_main_workspace(
        self,
        organism_name: str,
        strategy: str,
        constraints: List[OptimizationConstraint],
    ) -> None:
        """Render the main input/output workspace."""

        # Input section
        tab_paste, tab_upload = st.tabs(["📝 Paste Sequence", "📁 Upload FASTA"])

        sequences_to_optimize: List[dict] = []

        with tab_paste:
            seq_name = st.text_input(
                "Sequence Name",
                value="",
                placeholder="e.g., GFP, insulin, my_sequence",
                help="Optional name for your sequence.",
            )
            pasted = st.text_area(
                "Enter your sequence",
                height=200,
                placeholder=(
                    "Paste a protein or DNA sequence here...\n"
                    "The system will auto-detect whether it is DNA or protein.\n"
                    "FASTA format is also accepted."
                ),
            )
            if pasted.strip():
                parsed = self._parse_input(pasted.strip(), seq_name.strip())
                sequences_to_optimize.extend(parsed)

        with tab_upload:
            uploaded = st.file_uploader(
                "Upload a FASTA file",
                type=["fasta", "fa", "fna", "faa", "txt"],
            )
            if uploaded is not None:
                content = uploaded.read().decode("utf-8")
                parsed = self._parse_input(content)
                sequences_to_optimize.extend(parsed)

        # Auto-detect input type for each sequence
        for seq_info in sequences_to_optimize:
            seq_info["input_type"] = self._detect_sequence_type(seq_info["sequence"])

        # Show detected types
        if sequences_to_optimize:
            detected_types = set(s["input_type"] for s in sequences_to_optimize)
            label = ", ".join(t.upper() for t in sorted(detected_types))
            st.info(f"🔍 Auto-detected input type: **{label}**")

        # Optimize button
        if sequences_to_optimize and st.button(
            "🚀 Optimize", type="primary", use_container_width=True
        ):
            self._run_optimization(
                sequences_to_optimize, organism_name, strategy, constraints
            )

        # Show stored results
        if "results" in st.session_state and st.session_state.results:
            self._render_results(st.session_state.results)

    def _parse_input(self, text: str, name: str = "") -> List[dict]:
        """Parse raw text input into a list of sequences."""
        try:
            records = FastaParser.parse(text)
            result = []
            for r in records:
                # If the parser assigned a default name and the user provided one, use theirs
                record_name = r.name
                if record_name == "unnamed_sequence" and name:
                    record_name = name
                result.append(
                    {"name": record_name, "sequence": r.sequence, "description": r.description}
                )
            return result
        except ValueError:
            # Treat as raw sequence
            clean = text.replace("\n", "").replace(" ", "").upper()
            seq_name = name if name else "input_sequence"
            return [{"name": seq_name, "sequence": clean, "description": ""}]

    def _run_optimization(
        self,
        sequences: List[dict],
        organism_name: str,
        strategy: str,
        constraints: List[OptimizationConstraint],
    ) -> None:
        """Run optimization for all input sequences."""
        results: List[tuple[str, OptimizationResult | None, str]] = []

        for seq_info in sequences:
            name = seq_info["name"]
            sequence = seq_info["sequence"]
            input_type = seq_info.get("input_type", "protein")

            # Validate
            validation = self.service.validate_sequence(sequence, input_type)
            if not validation.is_valid:
                error_msg = f"Validation failed for '{name}': " + "; ".join(
                    validation.errors
                )
                results.append((name, None, error_msg))
                continue

            # Optimize
            try:
                result = self.service.optimize(
                    sequence=sequence,
                    input_type=input_type,
                    organism_name=organism_name,
                    strategy_name=strategy,
                    constraints=constraints,
                )
                # Attach validation warnings
                result.warnings = (validation.warnings or []) + (result.warnings or [])
                results.append((name, result, ""))
            except Exception as e:
                results.append((name, None, f"Optimization error for '{name}': {e}"))

        st.session_state.results = results

    def _render_results(
        self, results: List[tuple[str, OptimizationResult | None, str]]
    ) -> None:
        """Render optimization results."""
        st.header("📋 Results")

        for name, result, error in results:
            if error:
                st.error(error)
                continue

            if result is None:
                continue

            with st.expander(f"🧬 {name}", expanded=True):
                # Warnings
                render_warnings(result.warnings)

                # Metrics
                if result.metrics_before:
                    render_comparison_metrics(
                        result.metrics_before, result.metrics_after or {}
                    )
                else:
                    render_metrics_cards(result.metrics_after or {}, "Optimized Metrics")

                # Sequences
                col1, col2 = st.columns(2)
                with col1:
                    render_sequence_display(
                        result.protein_sequence,
                        f"Protein Sequence ({len(result.protein_sequence)} aa)",
                    )
                with col2:
                    render_sequence_display(
                        result.optimized_dna.sequence,
                        f"Optimized DNA ({len(result.optimized_dna)} bp)",
                    )

                # Codon usage chart
                render_codon_usage_chart(
                    result.optimized_dna.sequence,
                    title=f"Codon Usage – {name}",
                )

                # Export section
                self._render_export(name, result)

    def _render_export(self, name: str, result: OptimizationResult) -> None:
        """Render download buttons for exporting results."""
        st.subheader("💾 Export")
        col1, col2, col3 = st.columns(3)

        with col1:
            fasta_data = FastaExporter.export(result, name)
            st.download_button(
                "📄 Download FASTA",
                data=fasta_data,
                file_name=f"{name}_optimized.fasta",
                mime="text/plain",
            )
        with col2:
            csv_data = CsvExporter.export(result, name)
            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name=f"{name}_summary.csv",
                mime="text/csv",
            )
        with col3:
            text_data = TextExporter.export(result, name)
            st.download_button(
                "📝 Download Report",
                data=text_data,
                file_name=f"{name}_report.txt",
                mime="text/plain",
            )
