"""Tests for sequence validation and FASTA parsing."""

import pytest

from src.validation.validators import SequenceValidator
from src.validation.parsers import FastaParser


class TestDNAValidation:
    def test_valid_sequence(self):
        result = SequenceValidator.validate_dna("ATGAAAGCCTAA")
        assert result.is_valid
        assert not result.errors

    def test_empty_sequence(self):
        result = SequenceValidator.validate_dna("")
        assert not result.is_valid
        assert "empty" in result.errors[0].lower()

    def test_invalid_characters(self):
        result = SequenceValidator.validate_dna("ATGXYZ")
        assert not result.is_valid
        assert any("Invalid DNA" in e for e in result.errors)

    def test_not_divisible_by_3(self):
        result = SequenceValidator.validate_dna("ATGAA")
        assert not result.is_valid
        assert any("divisible by 3" in e for e in result.errors)

    def test_no_start_codon_warning(self):
        result = SequenceValidator.validate_dna("AAAGCCTAA")
        assert result.is_valid
        assert any("ATG" in w for w in result.warnings)

    def test_internal_stop_codon(self):
        result = SequenceValidator.validate_dna("ATGTAAATG")
        assert not result.is_valid
        assert any("stop codon" in e.lower() for e in result.errors)

    def test_trailing_stop_allowed(self):
        result = SequenceValidator.validate_dna("ATGAAATAA")
        assert result.is_valid


class TestProteinValidation:
    def test_valid_sequence(self):
        result = SequenceValidator.validate_protein("MKFLVDTY")
        assert result.is_valid

    def test_empty_sequence(self):
        result = SequenceValidator.validate_protein("")
        assert not result.is_valid

    def test_invalid_characters(self):
        result = SequenceValidator.validate_protein("MKFL123")
        assert not result.is_valid

    def test_trailing_stop(self):
        result = SequenceValidator.validate_protein("MKFLV*")
        assert result.is_valid

    def test_short_warning(self):
        result = SequenceValidator.validate_protein("M")
        assert result.is_valid
        assert len(result.warnings) > 0


class TestFastaParser:
    def test_single_record(self):
        text = ">seq1 test\nATGAAA\nGCCTAA"
        records = FastaParser.parse(text)
        assert len(records) == 1
        assert records[0].name == "seq1"
        assert records[0].sequence == "ATGAAAGCCTAA"

    def test_multiple_records(self):
        text = ">seq1\nATGAAA\n>seq2\nGCCTAA"
        records = FastaParser.parse(text)
        assert len(records) == 2
        assert records[0].sequence == "ATGAAA"
        assert records[1].sequence == "GCCTAA"

    def test_plain_sequence(self):
        text = "ATGAAAGCC"
        records = FastaParser.parse(text)
        assert len(records) == 1
        assert records[0].sequence == "ATGAAAGCC"

    def test_empty_input(self):
        with pytest.raises(ValueError, match="No valid FASTA"):
            FastaParser.parse("")

    def test_description(self):
        text = ">myseq A test sequence\nATGGCC"
        records = FastaParser.parse(text)
        assert records[0].name == "myseq"
        assert records[0].description == "A test sequence"
