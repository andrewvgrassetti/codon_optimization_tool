# 🧬 Codon Optimization Tool

A production-quality Streamlit application for codon optimization of DNA and protein sequences. Optimizes codon usage for target expression organisms using pluggable strategies and configurable constraints.

![App Screenshot](https://github.com/user-attachments/assets/b13ff4fd-9c5e-4625-b2ca-5fcdd98c8537)

## Features

- **Multiple input methods**: Paste sequences or upload FASTA files
- **DNA and protein input**: Accepts coding DNA sequences or amino acid sequences
- **4 built-in organisms**: E. coli (K12), S. cerevisiae, Human, CHO
- **Optimization strategies**:
  - *Highest Frequency*: Deterministic — always picks the most-used codon
  - *Weighted Random*: Stochastic — selects codons proportional to usage frequency
- **Configurable constraints**: GC content range, restriction site avoidance, homopolymer limits, custom motif exclusion
- **Comprehensive metrics**: CAI score (approximate), GC content (overall and per-position), codon distribution charts
- **Batch processing**: Optimize multiple sequences from a single FASTA file
- **Export**: Download results as FASTA, CSV summary, or plain text report

## Project Structure

```
codon_optimization_tool/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── data/
│   ├── codon_tables/
│   │   └── codon_usage.json        # Built-in organism codon usage data
│   └── examples/
│       └── sample_sequences.fasta  # Example FASTA input
├── src/
│   ├── models/
│   │   └── sequences.py            # BaseSequence, DNASequence, ProteinSequence
│   ├── config/
│   │   ├── constants.py            # Genetic code, amino acids, restriction sites
│   │   └── organisms.py            # OrganismProfile, CodonUsageTable, registry
│   ├── validation/
│   │   ├── validators.py           # SequenceValidator
│   │   └── parsers.py              # FastaParser
│   ├── optimization/
│   │   ├── strategies.py           # OptimizationStrategy, HighestFrequency, WeightedRandom
│   │   ├── constraints.py          # GC, restriction site, homopolymer, motif constraints
│   │   └── optimizer.py            # CodonOptimizer engine
│   ├── analysis/
│   │   └── metrics.py              # SequenceAnalyzer, CodonMetricsCalculator
│   ├── services/
│   │   └── optimization_service.py # OptimizationService (orchestration layer)
│   └── export/
│       └── exporters.py            # FASTA, CSV, text exporters
├── ui/
│   ├── components.py               # Reusable Streamlit UI components
│   └── app_controller.py           # StreamlitApp controller
└── tests/
    ├── test_models.py
    ├── test_validators.py
    ├── test_optimization.py
    ├── test_analysis.py
    └── test_services.py
```

## Setup

```bash
# Clone the repository
git clone https://github.com/andrewvgrassetti/codon_optimization_tool.git
cd codon_optimization_tool

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Running Tests

```bash
python -m pytest tests/ -v
```

## How the Optimization Works

### Codon Optimization

Every amino acid can be encoded by one or more DNA codons (synonymous codons). Different organisms prefer different codons for the same amino acid. This tool replaces each codon in a sequence with the preferred codon for the target expression host, preserving the amino acid sequence exactly.

### Strategies

1. **Highest Frequency**: For each amino acid, always select the single most-used codon in the target organism. This maximizes the CAI score but may produce repetitive sequences with extreme GC content.

2. **Weighted Random**: Select codons with probability proportional to their usage frequency. This produces more natural-looking sequences while still favoring preferred codons.

### CAI Score

The Codon Adaptation Index (CAI) is computed as the geometric mean of relative adaptiveness values for all codons. Relative adaptiveness is calculated as `frequency(codon) / max_frequency(amino_acid)`.

> **Note**: This is an approximation. A fully authoritative CAI requires reference gene sets from highly expressed genes, which is beyond the scope of this tool. The frequencies used here are genome-wide codon usage frequencies from the Kazusa Codon Usage Database.

### Constraints

Constraints are checked *after* optimization and reported as warnings:
- **GC Content**: Flags if overall GC% falls outside the configured range
- **Restriction Sites**: Detects common restriction enzyme recognition sequences
- **Homopolymer Runs**: Warns about long single-nucleotide repeats
- **Custom Motifs**: Flags user-specified DNA motifs

### Codon Usage Data

Built-in codon usage frequencies for 4 organisms are stored in `data/codon_tables/codon_usage.json`. Frequencies are per-amino-acid fractions (synonymous codons for each amino acid sum to 1.0). Data is sourced from published codon usage databases.

## Limitations

- Constraints are checked post-optimization (warning-based), not enforced during codon selection
- CAI is approximate (genome-wide frequencies, not reference gene sets)
- No codon pair optimization or mRNA secondary structure analysis
- No codon harmonization for heterologous expression

## Next Improvements

1. **Constraint enforcement during optimization**: Use iterative refinement or simulated annealing to satisfy constraints while optimizing
2. **Codon pair optimization**: Consider dicodon frequencies for improved translation efficiency
3. **mRNA structure prediction**: Integrate folding energy estimation (e.g., via ViennaRNA) to avoid problematic secondary structures
4. **More organisms**: Add organism profiles for insect cells (Sf9), Pichia pastoris, Bacillus subtilis, etc.
5. **Reference-set CAI**: Compute true CAI from highly expressed gene reference sets
6. **Codon harmonization**: Match codon usage patterns to the source organism's rare codon positions
7. **Multi-objective optimization**: Balance CAI, GC content, and constraint satisfaction using Pareto optimization
8. **REST API**: Add a FastAPI backend for programmatic access
9. **Persistent sessions**: Save and reload optimization sessions