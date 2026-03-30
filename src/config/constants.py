"""Genetic code constants and codon tables.

Uses the standard genetic code (NCBI translation table 1).
"""

from typing import Dict, List

# Standard genetic code: codon -> amino acid
CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS: List[str] = ["TAA", "TAG", "TGA"]

START_CODON: str = "ATG"

# Reverse mapping: amino acid -> list of codons
AMINO_ACID_TO_CODONS: Dict[str, List[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    if _aa != "*":
        AMINO_ACID_TO_CODONS.setdefault(_aa, []).append(_codon)

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"

# Single-letter amino acid code -> three-letter abbreviation
AMINO_ACID_ABBREVIATIONS: Dict[str, str] = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "*": "Stop",
}

VALID_DNA_BASES: str = "ATCG"

VALID_PROTEIN_CHARS: str = AMINO_ACIDS + "*"

# Common restriction enzyme recognition sites
COMMON_RESTRICTION_SITES: Dict[str, str] = {
    "AatII": "GACGTC",
    "AclI": "AACGTT",
    "AfeI": "AGCGCT",
    "AflII": "CTTAAG",
    "AgeI": "ACCGGT",
    "AluI": "AGCT",
    "ApaI": "GGGCCC",
    "ApaLI": "GTGCAC",
    "AscI": "GGCGCGCC",
    "AvrII": "CCTAGG",
    "BamHI": "GGATCC",
    "BbsI": "GAAGAC",
    "BclI": "TGATCA",
    "BglII": "AGATCT",
    "BsaI": "GGTCTC",
    "BseRI": "GAGGAG",
    "BsiWI": "CGTACG",
    "BsmBI": "CGTCTC",
    "BspEI": "TCCGGA",
    "BspQI": "GCTCTTC",
    "BsrGI": "TGTACA",
    "BssHII": "GCGCGC",
    "BstBI": "TTCGAA",
    "BstZ17I": "GTATAC",
    "ClaI": "ATCGAT",
    "DpnI": "GATC",
    "DraI": "TTTAAA",
    "EagI": "CGGCCG",
    "EcoRI": "GAATTC",
    "EcoRV": "GATATC",
    "FseI": "GGCCGGCC",
    "FspI": "TGCGCA",
    "HindIII": "AAGCTT",
    "HpaI": "GTTAAC",
    "KasI": "GGCGCC",
    "KpnI": "GGTACC",
    "MfeI": "CAATTG",
    "MluI": "ACGCGT",
    "MscI": "TGGCCA",
    "NaeI": "GCCGGC",
    "NcoI": "CCATGG",
    "NdeI": "CATATG",
    "NheI": "GCTAGC",
    "NotI": "GCGGCCGC",
    "NruI": "TCGCGA",
    "NsiI": "ATGCAT",
    "PacI": "TTAATTAA",
    "PciI": "ACATGT",
    "PmeI": "GTTTAAAC",
    "PmlI": "CACGTG",
    "PstI": "CTGCAG",
    "PvuI": "CGATCG",
    "PvuII": "CAGCTG",
    "SacI": "GAGCTC",
    "SacII": "CCGCGG",
    "SalI": "GTCGAC",
    "SbfI": "CCTGCAGG",
    "ScaI": "AGTACT",
    "SmaI": "CCCGGG",
    "SnaBI": "TACGTA",
    "SpeI": "ACTAGT",
    "SphI": "GCATGC",
    "SspI": "AATATT",
    "StuI": "AGGCCT",
    "SwaI": "ATTTAAAT",
    "XbaI": "TCTAGA",
    "XhoI": "CTCGAG",
    "XmaI": "CCCGGG",  # isoschizomer of SmaI (same recognition site, different cut position)
}
