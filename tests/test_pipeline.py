import json
from pathlib import Path

import numpy as np
import pytest

from causaldem_qec.artifacts import publish_trajectory
from causaldem_qec.cli import _execution_context, build_parser, main, select_pilot_partition
from causaldem_qec.core import (
    LabelTrajectory,
    ManifestProvenance,
    ObservableTrajectory,
    TrajectoryJob,
    expand_jobs,
    load_spec,
)
from causaldem_qec.report import DATASET_EDA_SECTIONS, build_dataset_eda, validate_inventory
from causaldem_qec.simulate import _manifest_payload, assert_run_manifest_identity


def _observable(start: int, rounds: int = 128) -> ObservableTrajectory:
    clock = np.arange(start, start + rounds, dtype=np.uint32)
    detector_bits = np.column_stack((clock % 2 == 0, clock % 3 == 0)).astype(np.bool_)
    detector_valid = np.ones_like(detector_bits)
    detector_valid[20:24, 0] = False
    detector_valid[72:80, 1] = False
    return ObservableTrajectory(
        detector_bits=detector_bits,
        detector_valid=detector_valid,
        logical_observable=np.zeros(rounds // 32, dtype=np.bool_),
        global_round=clock,
        episode=clock // 32,
        round_in_episode=clock % 32,
        block=clock // 256,
        detector_role=np.array([0, 1], dtype=np.uint16),
        circuit_phase=(clock % 32).astype(np.uint8),
        max_source_round=clock.astype(np.int64),
    )


def _labels(rounds: int = 128) -> LabelTrajectory:
    clock = np.arange(rounds, dtype=np.float64)
    component = np.column_stack((0.001 + clock / 100_000, 0.002 + clock / 200_000))
    classes = np.column_stack((0.01 + clock / 100_000, 0.02 + clock / 200_000))
    return LabelTrajectory(
        component_probability=component,
        latent_factor=np.zeros((rounds, 2), dtype=np.float64),
        class_probability=classes,
        future_block_probability=np.empty((0, 2), dtype=np.float64),
    )


@pytest.fixture
def tiny_published_pilot(tmp_path: Path) -> Path:
    spec = load_spec(Path("configs/poc_pilot.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "repetition_d3")
    root = tmp_path / "pilot"
    for trajectory_id, split in enumerate(("train", "development")):
        job = TrajectoryJob("repetition_d3__f01", trajectory_id, split, circuit, "f01", 713)
        publish_trajectory(
            root,
            job,
            _observable(trajectory_id * 128),
            _labels(),
            {
                "resolved_config_hash": "pilot-fixture",
                "dataset_profile": "pilot",
                "episode_rounds": 32,
                "block_rounds": 256,
                "canonical_catalog": {
                    "class_count": 2,
                    "duplicate_sizes": [1, 2],
                    "graphlike_mass": 0.02,
                    "adaptable_mass": 0.01,
                    "ambiguous_logical_mass": 0.0,
                    "hyperedge_mass": 0.0,
                },
                "generation_law": {
                    "component_bounds": [["repetition_data", 0.0001, 0.03]],
                    "missingness_parameters": {"mcar": 0.05, "mean_duration": 16},
                    "observation_flip_probability": 0.01,
                },
            },
        )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_profile": "pilot",
                "resolved_config_hash": "pilot-fixture",
                "results": [
                    {"condition_id": "repetition_d3__f01", "trajectory_id": item, "completed": True}
                    for item in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_dataset_eda_emits_every_pre_model_section_in_bounded_chunks(
    tiny_published_pilot: Path, tmp_path: Path
) -> None:
    index = build_dataset_eda(tiny_published_pilot, tmp_path, sample_seed=713, chunk_rounds=64)

    assert index.sections == DATASET_EDA_SECTIONS
    assert index.max_loaded_rounds <= 64
    assert index.truth_sections == {
        "physical_and_class_probabilities",
        "temporal_spectra",
        "spatial_correlations",
        "parity_theory_check",
        "noncommutation_gap",
        "canonical_catalog",
    }
    assert set(index.section_records) == set(DATASET_EDA_SECTIONS)
    assert all(record.output_path.exists() for record in index.section_records.values())
    assert "PILOT / NOT FINAL" in (tmp_path / "data_card.md").read_text(encoding="utf-8")
    assert "DQ08" in (tmp_path / "validation_report.json").read_text(encoding="utf-8")


def test_pilot_partition_selection_is_disjoint_and_covers_nonsealed_jobs() -> None:
    spec = load_spec(Path("configs/poc_pilot.json"))
    jobs = expand_jobs(spec, include_sealed=False)

    shards = [select_pilot_partition(spec, jobs, f"shard{index}") for index in range(1, 4)]

    assert [len(shard) for shard in shards] == [20, 20, 24]
    assert len({(job.condition_id, job.trajectory_id) for shard in shards for job in shard}) == 64
    assert {(job.condition_id, job.trajectory_id) for shard in shards for job in shard} == {
        (job.condition_id, job.trajectory_id) for job in jobs
    }
    assert all(job.split != "sealed_test" for shard in shards for job in shard)


def test_pilot_partition_selection_requires_pilot_config() -> None:
    spec = load_spec(Path("configs/poc.json"))

    with pytest.raises(ValueError, match="pilot configuration"):
        select_pilot_partition(spec, (), "shard1")


def test_bounded_generation_options_parse_and_flow_into_execution_context() -> None:
    args = build_parser().parse_args(
        [
            "generate-pilot",
            "--config",
            "configs/poc_pilot.json",
            "--generation-mode",
            "bounded",
            "--generation-chunk-rounds",
            "256",
        ]
    )

    options, provenance = _execution_context(args, load_spec(args.config))

    assert options.generation_mode == "bounded"
    assert options.generation_chunk_rounds == 256
    assert provenance is None


def test_bounded_generation_requires_a_chunk_size() -> None:
    args = build_parser().parse_args(
        [
            "generate-pilot",
            "--config",
            "configs/poc_pilot.json",
            "--generation-mode",
            "bounded",
        ]
    )

    with pytest.raises(ValueError, match="chunk rounds.*required"):
        _execution_context(args, load_spec(args.config))


def test_bounded_generation_chunk_size_must_align_to_episode_rounds() -> None:
    args = build_parser().parse_args(
        [
            "generate-pilot",
            "--config",
            "configs/poc_pilot.json",
            "--generation-mode",
            "bounded",
            "--generation-chunk-rounds",
            "250",
        ]
    )

    with pytest.raises(ValueError, match="multiple of episode_rounds"):
        _execution_context(args, load_spec(args.config))


def test_report_fails_on_duplicate_trajectory_id() -> None:
    with pytest.raises(ValueError, match="duplicate trajectory id"):
        validate_inventory([("repetition_d3__f01", 0), ("repetition_d3__f01", 0)])


def test_eda_sections_contain_stratified_dataset_evidence(
    tiny_published_pilot: Path, tmp_path: Path
) -> None:
    index = build_dataset_eda(tiny_published_pilot, tmp_path, sample_seed=713, chunk_rounds=64)
    section = lambda name: json.loads(index.output_paths[name].read_text(encoding="utf-8"))[
        "section_details"
    ]

    rates = section("detector_and_logical_rates")
    assert set(rates) >= {
        "by_circuit",
        "by_distance",
        "by_phase",
        "by_detector_role",
        "by_dynamics",
        "by_time_stratum",
        "logical_rate",
        "logical_by_circuit",
        "logical_by_distance",
        "logical_by_dynamics",
        "logical_by_time_stratum",
    }
    physical = section("physical_and_class_probabilities")
    assert set(physical) >= {
        "component_distribution",
        "class_distribution",
        "heterogeneity",
        "boundary_checks",
    }
    views = section("trajectory_views")
    assert set(views) >= {"long", "episode", "block", "burst", "regime"}
    corruption = section("observation_corruption")
    assert set(corruption) >= {
        "missing_run_lengths",
        "contamination",
        "burst_durations",
        "regime_dwell_times",
        "physical_independence",
    }
    catalog = section("canonical_catalog")
    assert catalog["class_counts"] == [2]
    assert catalog["duplicate_sizes"] == [1, 2]
    assert set(catalog) >= {"graphlike_mass", "adaptable_mass", "ambiguous_mass", "hyperedge_mass"}
    isolation = section("split_isolation")
    assert isolation["trajectory_ids_disjoint"] is True
    assert isolation["normalizer_isolation"]["fitted"] is False
    assert isolation["recurrent_state_isolation"]["fitted"] is False
    data_card = (tmp_path / "data_card.md").read_text(encoding="utf-8")
    assert "Geometry:" in data_card
    assert "Circuits:" in data_card
    assert "Physical-error bounds:" in data_card


def test_eda_cli_rejects_a_production_profile_and_incomplete_pilot(
    tiny_published_pilot: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="configs/poc_pilot.json"):
        main(
            [
                "eda-dataset",
                "--config",
                "configs/poc.json",
                "--output-root",
                str(tiny_published_pilot),
            ]
        )
    with pytest.raises(ValueError, match="complete pilot"):
        main(
            [
                "eda-dataset",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tiny_published_pilot),
                "--reports-root",
                str(tmp_path / "reports"),
            ]
        )


def test_eda_identity_preflight_accepts_a_valid_provenance_bound_kaggle_manifest(
    tmp_path: Path,
) -> None:
    spec = load_spec(Path("configs/poc_pilot.json"))
    provenance = ManifestProvenance(
        source_commit="task-3-review",
        execution_backend="kaggle",
        generation_law_version="standard_monolithic_v1",
        checkpoint_identity="owner/causaldem-pilot-checkpoint",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            _manifest_payload(
                {},
                spec,
                provenance=provenance,
                checkpoint_input_version="owner/causaldem-pilot-checkpoint@7",
            )
        ),
        encoding="utf-8",
    )

    assert_run_manifest_identity(
        tmp_path,
        spec,
        allow_bound_provenance=True,
    )


def test_eda_identity_preflight_rejects_non_kaggle_bound_provenance(tmp_path: Path) -> None:
    spec = load_spec(Path("configs/poc_pilot.json"))
    provenance = ManifestProvenance(
        source_commit="task-3-review",
        execution_backend="local",
        generation_law_version="standard_monolithic_v1",
        checkpoint_identity=None,
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(_manifest_payload({}, spec, provenance=provenance)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="execution identity"):
        assert_run_manifest_identity(
            tmp_path,
            spec,
            allow_bound_provenance=True,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["generate-pilot", "--execution-backend", "local", "--job-limit", "1"],
        ["generate-pilot", "--execution-backend", "kaggle"],
        [
            "generate-pilot",
            "--execution-backend",
            "kaggle",
            "--job-limit",
            "1",
            "--checkpoint-root",
            "checkpoint",
            "--workers",
            "2",
        ],
        [
            "generate",
            "--execution-backend",
            "kaggle",
            "--job-limit",
            "1",
            "--checkpoint-root",
            "checkpoint",
        ],
    ],
)
def test_invalid_kaggle_cli_combinations_fail_before_output(
    arguments: list[str], tmp_path: Path
) -> None:
    output_root = tmp_path / "output"
    with pytest.raises(ValueError):
        main(
            [
                *arguments,
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
    assert not output_root.exists()


def test_kaggle_pilot_dry_run_accepts_explicit_bounded_checkpoint_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "generate-pilot",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tmp_path / "run"),
                "--execution-backend",
                "kaggle",
                "--job-limit",
                "1",
                "--checkpoint-root",
                str(tmp_path / "checkpoint"),
                "--checkpoint-identity",
                "owner/causaldem-pilot-checkpoint",
                "--checkpoint-version",
                "owner/causaldem-pilot-checkpoint@7",
                "--dry-run",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["execution_backend"] == "kaggle"
    assert status["job_limit"] == 1
    assert status["checkpoint_root"] == str((tmp_path / "checkpoint").resolve())
    assert status["checkpoint_identity"] == "owner/causaldem-pilot-checkpoint"
    assert status["checkpoint_input_version"] == "owner/causaldem-pilot-checkpoint@7"
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "checkpoint").exists()


def test_kaggle_cli_requires_external_checkpoint_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint identity"):
        main(
            [
                "generate-pilot",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tmp_path / "run"),
                "--execution-backend",
                "kaggle",
                "--job-limit",
                "1",
                "--checkpoint-root",
                str(tmp_path / "checkpoint"),
                "--dry-run",
            ]
        )
    assert not (tmp_path / "run").exists()


def test_kaggle_cli_rejects_checkpoint_version_from_another_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint version.*identity"):
        main(
            [
                "generate-pilot",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tmp_path / "run"),
                "--execution-backend",
                "kaggle",
                "--job-limit",
                "1",
                "--checkpoint-root",
                str(tmp_path / "checkpoint"),
                "--checkpoint-identity",
                "owner/causaldem-pilot-checkpoint",
                "--checkpoint-version",
                "other-owner/other-checkpoint@7",
                "--dry-run",
            ]
        )
    assert not (tmp_path / "run").exists()


def test_kaggle_cli_rejects_checkpoint_overlap_with_private_manifest(tmp_path: Path) -> None:
    private = tmp_path / "checkpoint" / "sealed.json"
    with pytest.raises(ValueError, match="private sealed manifest"):
        main(
            [
                "generate-pilot",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tmp_path / "run"),
                "--execution-backend",
                "kaggle",
                "--job-limit",
                "1",
                "--checkpoint-root",
                str(tmp_path / "checkpoint"),
                "--checkpoint-identity",
                "owner/causaldem-pilot-checkpoint",
                "--checkpoint-version",
                "owner/causaldem-pilot-checkpoint@7",
                "--sealed-manifest",
                str(private),
                "--dry-run",
            ]
        )
    assert not (tmp_path / "run").exists()


def test_notebook_is_presentation_only() -> None:
    notebook = json.loads(Path("notebooks/eda.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 1
    source = "".join(code_cells[0]["source"])
    assert "build_dataset_eda(" in source
    assert "display_eda(index)" in source
