"""Sequence validation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.config.constants import CODON_TABLE, VALID_DNA_BASES, VALID_PROTEIN_CHARS


@dataclass
class ValidationResult:
    """Result of a sequence validation check."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SequenceValidator:
    """Validates DNA and protein sequences."""

    @staticmethod
    def validate_dna(sequence: str) -> ValidationResult:
        """Validate a DNA coding sequence.

        Checks:
        - Non-empty
        - Contains only valid DNA bases (A, T, C, G)
        - Length is divisible by 3
        - Starts with ATG (warning if not)
        - All codons are valid
        """
        errors: List[str] = []
        warnings: List[str] = []
        seq = sequence.upper().strip()

        if not seq:
            return ValidationResult(is_valid=False, errors=["Sequence is empty."])

        # Check for invalid characters
        invalid_chars = set(seq) - set(VALID_DNA_BASES)
        if invalid_chars:
            errors.append(
                f"Invalid DNA characters found: {', '.join(sorted(invalid_chars))}. "
                f"Only A, T, C, G are allowed."
            )

        if len(seq) % 3 != 0:
            errors.append(
                f"Sequence length ({len(seq)}) is not divisible by 3. "
                f"This is not a valid coding sequence."
            )

        if not errors:
            if not seq.startswith("ATG"):
                warnings.append(
                    "Sequence does not start with ATG (standard start codon)."
                )

            # Check for internal stop codons (excluding the last codon)
            for i in range(0, len(seq) - 3, 3):
                codon = seq[i : i + 3]
                aa = CODON_TABLE.get(codon)
                if aa == "*":
                    errors.append(
                        f"Internal stop codon ({codon}) found at position {i + 1}."
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def validate_protein(sequence: str) -> ValidationResult:
        """Validate a protein sequence.

        Checks:
        - Non-empty
        - Contains only valid amino acid characters
        """
        errors: List[str] = []
        warnings: List[str] = []
        seq = sequence.upper().strip()

        if not seq:
            return ValidationResult(is_valid=False, errors=["Sequence is empty."])

        # Remove trailing stop if present
        if seq.endswith("*"):
            seq = seq[:-1]

        invalid_chars = set(seq) - set(VALID_PROTEIN_CHARS)
        if invalid_chars:
            errors.append(
                f"Invalid protein characters found: {', '.join(sorted(invalid_chars))}. "
                f"Valid amino acids: {VALID_PROTEIN_CHARS}"
            )

        if not errors and len(seq) < 2:
            warnings.append("Sequence is very short (less than 2 amino acids).")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
