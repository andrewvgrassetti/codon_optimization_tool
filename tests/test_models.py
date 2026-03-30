"""Tests for sequence models."""

import pytest

from src.models.sequences import DNASequence, ProteinSequence, BaseSequence, VariantConfig


class TestBaseSequence:
    def test_uppercase_and_strip(self):
        seq = BaseSequence(sequence="  atcg  ")
        assert seq.sequence == "ATCG"

    def test_len(self):
        seq = BaseSequence(sequence="ATCG")
        assert len(seq) == 4

    def test_str(self):
        seq = BaseSequence(sequence="ATCG")
        assert str(seq) == "ATCG"


class TestDNASequence:
    def test_translate_simple(self):
        # ATG=M, AAA=K, TAA=stop
        dna = DNASequence(sequence="ATGAAATAA")
        assert dna.translate() == "MK"

    def test_translate_no_stop(self):
        dna = DNASequence(sequence="ATGAAA")
        assert dna.translate() == "MK"

    def test_translate_invalid_length(self):
        dna = DNASequence(sequence="ATGA")
        with pytest.raises(ValueError, match="not divisible by 3"):
            dna.translate()

    def test_gc_content(self):
        dna = DNASequence(sequence="GCGC")
        assert dna.gc_content == 1.0
        dna2 = DNASequence(sequence="ATAT")
        assert dna2.gc_content == 0.0
        dna3 = DNASequence(sequence="ATGC")
        assert dna3.gc_content == 0.5

    def test_gc_content_empty(self):
        dna = DNASequence(sequence="")
        assert dna.gc_content == 0.0

    def test_get_codons(self):
        dna = DNASequence(sequence="ATGAAAGCCTAA")
        codons = dna.get_codons()
        assert codons == ["ATG", "AAA", "GCC"]

    def test_get_codons_no_stop(self):
        dna = DNASequence(sequence="ATGAAAGCC")
        codons = dna.get_codons()
        assert codons == ["ATG", "AAA", "GCC"]


class TestProteinSequence:
    def test_creation(self):
        prot = ProteinSequence(sequence="MKFLV")
        assert prot.sequence == "MKFLV"
        assert len(prot) == 5


class TestVariantConfig:
    def test_default_values(self):
        config = VariantConfig()
        assert config.strategy_name == "highest_frequency"
        assert config.gc_min is None
        assert config.gc_max is None

    def test_label_highest_frequency(self):
        config = VariantConfig(strategy_name="highest_frequency")
        assert config.label == "Highest Frequency"

    def test_label_weighted_random(self):
        config = VariantConfig(strategy_name="weighted_random")
        assert config.label == "Weighted Random"

    def test_label_with_gc_range(self):
        config = VariantConfig(
            strategy_name="weighted_random", gc_min=0.40, gc_max=0.60
        )
        assert "Weighted Random" in config.label
        assert "GC 40%–60%" in config.label

    def test_label_without_gc_range(self):
        config = VariantConfig(strategy_name="highest_frequency")
        assert "GC" not in config.label
