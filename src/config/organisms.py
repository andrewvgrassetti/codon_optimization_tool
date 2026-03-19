"""Organism codon usage profiles and registry.

Codon usage frequencies are expressed as fractions within each amino acid group
(i.e., for each amino acid, the frequencies of its synonymous codons sum to 1.0).

Data sourced from the Codon Usage Database (Kazusa) and published literature.
These are representative values for common expression hosts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config.constants import AMINO_ACID_TO_CODONS


@dataclass
class CodonUsageTable:
    """Codon usage frequency table for a specific organism.

    Frequencies are per-amino-acid fractions (sum to 1.0 for each amino acid).
    """

    frequencies: Dict[str, float] = field(default_factory=dict)

    def get_frequency(self, codon: str) -> float:
        """Get the usage frequency for a codon."""
        return self.frequencies.get(codon.upper(), 0.0)

    def get_codons_for_amino_acid(self, amino_acid: str) -> Dict[str, float]:
        """Return {codon: frequency} for all codons encoding the given amino acid."""
        codons = AMINO_ACID_TO_CODONS.get(amino_acid.upper(), [])
        return {c: self.get_frequency(c) for c in codons}

    def get_best_codon(self, amino_acid: str) -> str:
        """Return the highest-frequency codon for a given amino acid."""
        codon_freqs = self.get_codons_for_amino_acid(amino_acid)
        if not codon_freqs:
            raise ValueError(f"No codons found for amino acid: {amino_acid}")
        return max(codon_freqs, key=codon_freqs.get)  # type: ignore[arg-type]


@dataclass
class OrganismProfile:
    """Profile for a target expression organism."""

    name: str
    display_name: str
    codon_table: CodonUsageTable
    description: str = ""

    def __str__(self) -> str:
        return self.display_name


class OrganismRegistry:
    """Registry of available organism profiles.

    Supports loading built-in profiles and registering custom ones.
    """

    def __init__(self) -> None:
        self._organisms: Dict[str, OrganismProfile] = {}

    def register(self, profile: OrganismProfile) -> None:
        """Register an organism profile."""
        self._organisms[profile.name] = profile

    def get(self, name: str) -> Optional[OrganismProfile]:
        """Get an organism profile by internal name."""
        return self._organisms.get(name)

    def list_organisms(self) -> List[OrganismProfile]:
        """List all registered organism profiles."""
        return list(self._organisms.values())

    def list_names(self) -> List[str]:
        """List internal names of all registered organisms."""
        return list(self._organisms.keys())

    def load_from_json(self, filepath: str | Path) -> None:
        """Load organism profiles from a JSON file."""
        path = Path(filepath)
        with open(path) as f:
            data = json.load(f)

        for org_key, org_data in data.items():
            table = CodonUsageTable(frequencies=org_data["codon_frequencies"])
            profile = OrganismProfile(
                name=org_key,
                display_name=org_data.get("display_name", org_key),
                codon_table=table,
                description=org_data.get("description", ""),
            )
            self.register(profile)


def get_default_registry() -> OrganismRegistry:
    """Create and return the default organism registry loaded from built-in data."""
    registry = OrganismRegistry()
    data_path = Path(__file__).parent.parent.parent / "data" / "codon_tables" / "codon_usage.json"
    if data_path.exists():
        registry.load_from_json(data_path)
    return registry
