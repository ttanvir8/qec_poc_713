import json
from pathlib import Path

import numpy as np
import pytest

from causaldem_qec.artifacts import publish_trajectory
from causaldem_qec.cli import main
from causaldem_qec.core import LabelTrajectory, ObservableTrajectory, TrajectoryJob, load_spec
from causaldem_qec.report import DATASET_EDA_SECTIONS, build_dataset_eda, validate_inventory


def _observable(start: int, rounds: int = 128) -> ObservableTrajectory:
    clock = np.arange(start, start + rounds, dtype=np.uint32)
    detector_bits = np.column_stack((clock % 2 == 0, clock % 3 == 0)).astype(np.bool_)
    return ObservableTrajectory(
        detector_bits=detector_bits,
        detector_valid=np.ones_like(detector_bits),
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
    component = np.tile(np.array([0.001, 0.002], dtype=np.float64), (rounds, 1))
    classes = np.tile(np.array([0.01, 0.02], dtype=np.float64), (rounds, 1))
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
            {"resolved_config_hash": "pilot-fixture", "dataset_profile": "pilot"},
        )
    (root / "run_manifest.json").write_text(
        json.dumps({"dataset_profile": "pilot", "resolved_config_hash": "pilot-fixture"}),
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


def test_report_fails_on_duplicate_trajectory_id() -> None:
    with pytest.raises(ValueError, match="duplicate trajectory id"):
        validate_inventory([("repetition_d3__f01", 0), ("repetition_d3__f01", 0)])


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


def test_notebook_is_presentation_only() -> None:
    notebook = json.loads(Path("notebooks/eda.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 1
    source = "".join(code_cells[0]["source"])
    assert "build_dataset_eda(" in source
    assert "display_eda(index)" in source
