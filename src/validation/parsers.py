"""FASTA file parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FastaRecord:
    """A single FASTA record with header and sequence."""

    header: str
    sequence: str

    @property
    def name(self) -> str:
        """Extract the sequence name from the FASTA header."""
        return self.header.split()[0] if self.header else ""

    @property
    def description(self) -> str:
        """Extract the description from the FASTA header."""
        parts = self.header.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""


class FastaParser:
    """Parse FASTA-formatted text into records."""

    @staticmethod
    def parse(text: str) -> List[FastaRecord]:
        """Parse FASTA-formatted text.

        Args:
            text: FASTA-formatted string (one or more records).

        Returns:
            List of FastaRecord objects.

        Raises:
            ValueError: If the text is not valid FASTA format.
        """
        records: List[FastaRecord] = []
        current_header: str | None = None
        current_seq_parts: List[str] = []

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous record
                if current_header is not None:
                    records.append(
                        FastaRecord(
                            header=current_header,
                            sequence="".join(current_seq_parts).upper(),
                        )
                    )
                current_header = line[1:].strip()
                current_seq_parts = []
            else:
                if current_header is None:
                    # Text without a header — treat as a plain sequence
                    current_header = "unnamed_sequence"
                current_seq_parts.append(line.replace(" ", ""))

        # Save last record
        if current_header is not None:
            records.append(
                FastaRecord(
                    header=current_header,
                    sequence="".join(current_seq_parts).upper(),
                )
            )

        if not records:
            raise ValueError("No valid FASTA records found in the input.")

        return records
