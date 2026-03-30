"""Tests for UI helper functions: auto-detection and codon labeling."""

import pytest

from ui.app_controller import StreamlitApp
from ui.components import _codon_label


class TestSequenceTypeAutoDetection:
    """Test the auto-detection of DNA vs. protein sequences."""

    def test_detect_dna_standard(self):
        assert StreamlitApp._detect_sequence_type("ATGAAAGCCTAA") == "dna"

    def test_detect_dna_lowercase(self):
        assert StreamlitApp._detect_sequence_type("atgaaagcctaa") == "dna"

    def test_detect_dna_not_divisible_by_3(self):
        # All ATCG but not divisible by 3 -> still DNA (validator will flag)
        assert StreamlitApp._detect_sequence_type("ATGAA") == "dna"

    def test_detect_protein_with_non_dna_chars(self):
        assert StreamlitApp._detect_sequence_type("MKFLVDTY") == "protein"

    def test_detect_protein_single_letter(self):
        assert StreamlitApp._detect_sequence_type("M") == "protein"

    def test_detect_empty_sequence(self):
        assert StreamlitApp._detect_sequence_type("") == "protein"

    def test_detect_protein_with_mixed_chars(self):
        assert StreamlitApp._detect_sequence_type("MHWDEKRQ") == "protein"

    def test_detect_dna_all_atcg_div3(self):
        assert StreamlitApp._detect_sequence_type("ATCGATCGATCG") == "dna"

    def test_detect_ambiguous_act_protein(self):
        # "ACT" is only ATCG, length 3 -> detected as DNA
        assert StreamlitApp._detect_sequence_type("ACT") == "dna"

    def test_detect_protein_long_sequence(self):
        protein = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLD"
        assert StreamlitApp._detect_sequence_type(protein) == "protein"


class TestCodonLabel:
    """Test the codon labeling with amino acid abbreviation."""

    def test_atg_label(self):
        assert _codon_label("ATG") == "ATG (Met)"

    def test_ttt_label(self):
        assert _codon_label("TTT") == "TTT (Phe)"

    def test_ctg_label(self):
        assert _codon_label("CTG") == "CTG (Leu)"

    def test_tgg_label(self):
        assert _codon_label("TGG") == "TGG (Trp)"

    def test_unknown_codon_no_parentheses(self):
        assert _codon_label("XYZ") == "XYZ"

    def test_stop_codon_label(self):
        assert _codon_label("TAA") == "TAA (Stop)"

    def test_lowercase_codon(self):
        assert _codon_label("atg") == "atg (Met)"


class TestCodonTableSource:
    """Verify that codon usage data has GenScript source attribution."""

    def test_json_has_source_field(self):
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "data" / "codon_tables" / "codon_usage.json"
        with open(path) as f:
            data = json.load(f)
        for org_key, org_data in data.items():
            assert "source" in org_data, f"Missing source for {org_key}"
            assert "GenScript" in org_data["source"], f"Source not GenScript for {org_key}"
