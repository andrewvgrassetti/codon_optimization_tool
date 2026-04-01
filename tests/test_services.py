"""Tests for the high-level optimization service."""

import pytest

from src.services.optimization_service import OptimizationService


@pytest.fixture
def service():
    return OptimizationService()


class TestOptimizationService:
    def test_get_organisms(self, service):
        organisms = service.get_organisms()
        assert len(organisms) >= 4
        names = [o.name for o in organisms]
        assert "e_coli" in names
        assert "human" in names

    def test_validate_dna_valid(self, service):
        result = service.validate_sequence("ATGAAAGCCTAA", "dna")
        assert result.is_valid

    def test_validate_protein_valid(self, service):
        result = service.validate_sequence("MKFLV", "protein")
        assert result.is_valid

    def test_optimize_protein(self, service):
        result = service.optimize(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
        )
        assert result.optimized_dna is not None
        assert result.protein_sequence == "MKFLV"
        assert result.optimized_dna.translate() == "MKFLV"
        assert result.metrics_after

    def test_optimize_dna(self, service):
        result = service.optimize(
            sequence="ATGAAATTTCTGGTGTAA",
            input_type="dna",
            organism_name="e_coli",
        )
        assert result.optimized_dna is not None
        assert result.metrics_before
        assert result.metrics_after

    def test_optimize_preserves_protein_from_dna(self, service):
        from src.models.sequences import DNASequence

        dna = "ATGAAATTTCTGGTGTAA"
        original_protein = DNASequence(sequence=dna).translate()
        result = service.optimize(
            sequence=dna,
            input_type="dna",
            organism_name="human",
        )
        assert result.optimized_dna.translate() == original_protein

    def test_optimize_unknown_organism(self, service):
        with pytest.raises(ValueError, match="Unknown organism"):
            service.optimize(
                sequence="MKFLV",
                input_type="protein",
                organism_name="martian",
            )

    def test_optimize_with_constraints(self, service):
        from src.optimization.constraints import GCContentConstraint

        result = service.optimize(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            constraints=[GCContentConstraint(min_gc=0.90, max_gc=1.0)],
        )
        # The constraint should produce a warning since GC can't be 90%
        assert any("GC content" in w for w in result.warnings)

    def test_weighted_random_strategy(self, service):
        result = service.optimize(
            sequence="MKFLVDTY",
            input_type="protein",
            organism_name="e_coli",
            strategy_name="weighted_random",
            seed=42,
        )
        assert result.optimized_dna.translate() == "MKFLVDTY"


class TestOptimizeVariants:
    """Tests for multi-variant optimization."""

    def test_single_variant_returns_one_result(self, service):
        from src.models.sequences import VariantConfig

        configs = [VariantConfig(strategy_name="highest_frequency")]
        results = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 1
        assert results[0].optimized_dna.translate() == "MKFLV"
        assert results[0].variant_label == "Variant 1 – Highest Frequency"
        assert results[0].metrics_after

    def test_multiple_variants_returns_correct_count(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(strategy_name="highest_frequency"),
            VariantConfig(strategy_name="weighted_random"),
            VariantConfig(strategy_name="weighted_random"),
        ]
        results = service.optimize_variants(
            sequence="MKFLVDTY",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 3
        # All variants must preserve the protein sequence
        for r in results:
            assert r.optimized_dna.translate() == "MKFLVDTY"
            assert r.metrics_after

    def test_different_strategies_may_produce_different_sequences(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(strategy_name="highest_frequency"),
            VariantConfig(strategy_name="weighted_random"),
        ]
        results = service.optimize_variants(
            sequence="MKFLVDTYWSCRHP",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 2
        # Both preserve the protein
        for r in results:
            assert r.optimized_dna.translate() == "MKFLVDTYWSCRHP"
        # They should have different variant labels
        assert "Highest Frequency" in results[0].variant_label
        assert "Weighted Random" in results[1].variant_label

    def test_variant_with_gc_constraint(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(strategy_name="highest_frequency", gc_min=0.90, gc_max=1.0),
        ]
        results = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 1
        # GC constraint should trigger a warning
        assert any("GC content" in w for w in results[0].warnings)
        assert "GC 90%–100%" in results[0].variant_label

    def test_variants_with_different_gc_constraints(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(strategy_name="highest_frequency", gc_min=0.30, gc_max=0.70),
            VariantConfig(strategy_name="highest_frequency", gc_min=0.90, gc_max=1.0),
        ]
        results = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 2
        # Second variant should warn about GC; first may not
        assert any("GC content" in w for w in results[1].warnings)

    def test_shared_constraints_applied_to_all_variants(self, service):
        from src.models.sequences import VariantConfig
        from src.optimization.constraints import RestrictionSiteConstraint

        shared = [RestrictionSiteConstraint(sites_to_avoid={"EcoRI": "GAATTC"})]
        configs = [
            VariantConfig(strategy_name="highest_frequency"),
            VariantConfig(strategy_name="weighted_random"),
        ]
        results = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
            shared_constraints=shared,
        )
        assert len(results) == 2
        # Both variants should have been checked for restriction sites
        for r in results:
            assert r.optimized_dna.translate() == "MKFLV"

    def test_variant_from_dna_input(self, service):
        from src.models.sequences import DNASequence, VariantConfig

        dna = "ATGAAATTTCTGGTGTAA"
        original_protein = DNASequence(sequence=dna).translate()
        configs = [
            VariantConfig(strategy_name="highest_frequency"),
            VariantConfig(strategy_name="weighted_random"),
        ]
        results = service.optimize_variants(
            sequence=dna,
            input_type="dna",
            organism_name="human",
            variant_configs=configs,
        )
        assert len(results) == 2
        for r in results:
            assert r.optimized_dna.translate() == original_protein
            assert r.metrics_before
            assert r.metrics_after

    def test_variant_with_wrscu_constraint(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(
                strategy_name="highest_frequency",
                wrscu_min=0.90,
                wrscu_max=1.10,
            ),
        ]
        results = service.optimize_variants(
            sequence="MKFLVDTY",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 1
        # Highest-frequency strategy typically produces wRSCU below 0.90
        assert any("wRSCU" in w for w in results[0].warnings)
        assert "wRSCU 0.90–1.10" in results[0].variant_label

    def test_variant_with_gc_and_wrscu_constraints(self, service):
        from src.models.sequences import VariantConfig

        configs = [
            VariantConfig(
                strategy_name="highest_frequency",
                gc_min=0.90,
                gc_max=1.00,
                wrscu_min=0.90,
                wrscu_max=1.10,
            ),
        ]
        results = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        assert len(results) == 1
        # Both constraints should trigger warnings
        assert any("GC content" in w for w in results[0].warnings)
        assert any("wRSCU" in w for w in results[0].warnings)


class TestMultiVariantCsvExporter:
    """Tests for the consolidated multi-variant CSV exporter."""

    def test_csv_headers(self, service):
        from src.export.exporters import MultiVariantCsvExporter
        from src.models.sequences import VariantConfig
        import csv
        import io

        configs = [VariantConfig(strategy_name="highest_frequency")]
        results_raw = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        results = [("test_seq", results_raw[0], "")]
        csv_str = MultiVariantCsvExporter.export(results)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == [
            "name", "optimization strategy", "%GC range",
            "%GC of CDS", "%GC 1", "%GC 2", "%GC 3",
            "CAI", "wRSCU", "sequence",
        ]

    def test_csv_multiple_rows(self, service):
        from src.export.exporters import MultiVariantCsvExporter
        from src.models.sequences import VariantConfig
        import csv
        import io

        configs = [
            VariantConfig(strategy_name="highest_frequency"),
            VariantConfig(strategy_name="weighted_random"),
        ]
        results_raw = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        results = [
            ("seq_v1", results_raw[0], ""),
            ("seq_v2", results_raw[1], ""),
        ]
        csv_str = MultiVariantCsvExporter.export(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[1][0] == "seq_v1"
        assert rows[2][0] == "seq_v2"
        # Check strategy column
        assert rows[1][1] == "Highest Frequency"
        assert rows[2][1] == "Weighted Random"
        # Check sequence column is non-empty
        assert len(rows[1][9]) > 0
        assert len(rows[2][9]) > 0

    def test_csv_skips_errors(self, service):
        from src.export.exporters import MultiVariantCsvExporter
        from src.models.sequences import VariantConfig
        import csv
        import io

        configs = [VariantConfig(strategy_name="highest_frequency")]
        results_raw = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        results = [
            ("good_seq", results_raw[0], ""),
            ("bad_seq", None, "Some error"),
        ]
        csv_str = MultiVariantCsvExporter.export(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row (error skipped)

    def test_csv_gc_range_in_output(self, service):
        from src.export.exporters import MultiVariantCsvExporter
        from src.models.sequences import VariantConfig
        import csv
        import io

        configs = [
            VariantConfig(
                strategy_name="highest_frequency",
                gc_min=0.40,
                gc_max=0.60,
            ),
        ]
        results_raw = service.optimize_variants(
            sequence="MKFLV",
            input_type="protein",
            organism_name="e_coli",
            variant_configs=configs,
        )
        results = [("test_seq", results_raw[0], "")]
        csv_str = MultiVariantCsvExporter.export(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # The GC range column should contain the range
        assert "40%–60%" in rows[1][2]
