import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import causaldem_qec.artifacts as artifact_module
from causaldem_qec.artifacts import (
    load_labels,
    load_observable,
    publish_trajectory,
    verify_artifact,
)
from causaldem_qec.core import (
    CircuitSpec,
    LabelTrajectory,
    ObservableTrajectory,
    TrajectoryJob,
    derive_seed,
    expand_jobs,
    load_spec,
    source_cutoff,
    target_interval,
    validate_observable,
)

CONFIG = Path("configs/poc.json")


def _modified_config(
    tmp_path: Path, section: str, key: str, value: object, *, nested_key: str | None = None
) -> Path:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if nested_key is None:
        config[section][key] = value
    else:
        config[section][key][nested_key] = value
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_expands_exact_committed_matrix() -> None:
    spec = load_spec(CONFIG)
    jobs = expand_jobs(spec, include_sealed=True)
    assert len({job.condition_id for job in jobs}) == 30
    assert len(jobs) == 1920
    assert sum(job.split == "train" for job in jobs) == 320
    assert sum(job.split == "validation" for job in jobs) == 160
    assert sum(job.split == "id_test" for job in jobs) == 160


def test_resolved_manifest_has_exact_trajectory_and_round_totals() -> None:
    spec = load_spec(CONFIG)
    jobs = expand_jobs(spec, include_sealed=True)
    assert sum(job.split == "development" for job in jobs) == 512
    assert sum(job.split == "sealed_test" for job in jobs) == 768
    assert len(jobs) * spec.burn_in_rounds == 7_864_320
    assert len(jobs) * spec.scored_rounds == 125_829_120
    assert len(jobs) * (spec.scored_rounds // spec.episode_rounds) == 3_932_160
    assert len(jobs) * (spec.scored_rounds // spec.block_rounds) == 491_520


def test_semantic_seed_does_not_depend_on_call_order() -> None:
    first = derive_seed(713, "surface_d3", "f03", 7, "dynamics").generate_state(8)
    derive_seed(713, "unused").generate_state(8)
    second = derive_seed(713, "surface_d3", "f03", 7, "dynamics").generate_state(8)
    np.testing.assert_array_equal(first, second)


def test_block_b_forecasts_exactly_block_b_plus_two() -> None:
    assert source_cutoff(block=4, block_rounds=256) == 1279
    assert target_interval(block=4, block_rounds=256) == (1536, 1792)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "unknown": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_spec(path)


@pytest.mark.parametrize(
    ("dynamics_id", "key"),
    [
        ("f07", "mcar"),
        ("f07", "burst_hazard"),
        ("f07", "detector_fraction"),
        ("f08", "flip_probability"),
        ("f12", "onset_hazard"),
    ],
)
def test_probabilities_and_hazards_reject_negative_values(
    tmp_path: Path, dynamics_id: str, key: str
) -> None:
    path = _modified_config(tmp_path, "dynamics", dynamics_id, -0.001, nested_key=key)
    with pytest.raises(ValueError, match="outside its valid range"):
        load_spec(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("burn_in", 8192), ("scored", 131072), ("episode", 64), ("block", 512)],
)
def test_round_contract_rejects_noncommitted_fidelity_values(
    tmp_path: Path, key: str, value: int
) -> None:
    path = _modified_config(tmp_path, "rounds", key, value)
    with pytest.raises(ValueError, match="committed trajectory fidelity"):
        load_spec(path)


def test_retry_count_rejects_more_than_three_attempts(tmp_path: Path) -> None:
    path = _modified_config(tmp_path, "runtime", "retry_attempts", 4)
    with pytest.raises(ValueError, match="retry_attempts must not exceed 3"):
        load_spec(path)


@pytest.fixture
def tiny_job() -> TrajectoryJob:
    return TrajectoryJob(
        condition_id="repetition_d3__f01",
        trajectory_id=0,
        split="train",
        circuit=CircuitSpec("repetition_d3", "repetition", 3),
        dynamics_id="f01",
        root_seed=713,
    )


def _observable(rounds: int = 64, detectors: int = 3) -> ObservableTrajectory:
    bits = np.zeros((rounds, detectors), dtype=np.bool_)
    index = np.arange(rounds, dtype=np.uint32)
    return ObservableTrajectory(
        detector_bits=bits,
        detector_valid=np.ones_like(bits),
        logical_observable=np.zeros(rounds // 32, dtype=np.bool_),
        global_round=index,
        episode=index // 32,
        round_in_episode=index % 32,
        block=index // 256,
        detector_role=np.arange(detectors, dtype=np.uint16),
        circuit_phase=(index % 32).astype(np.uint8),
        max_source_round=index.astype(np.int64),
    )


def _labels(rounds: int = 64, classes: int = 2) -> LabelTrajectory:
    probability = np.full((rounds, classes), 0.01, dtype=np.float64)
    return LabelTrajectory(
        component_probability=probability,
        latent_factor=np.zeros((rounds, 2), dtype=np.float64),
        class_probability=probability.copy(),
        future_block_probability=np.empty((0, classes), dtype=np.float64),
    )


def test_observable_and_labels_publish_to_separate_hashes(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable = _observable()
    labels = _labels()
    observable_path, label_path = publish_trajectory(
        tmp_path, tiny_job, observable, labels, {"schema_version": 1}
    )
    assert observable_path.parts[-4] == "observable"
    assert label_path.parts[-4] == "labels"
    assert verify_artifact(observable_path) != verify_artifact(label_path)
    assert load_observable(observable_path).detector_bits.dtype == np.bool_
    assert (
        load_labels(label_path, purpose="offline_evaluation").class_probability.dtype == np.float64
    )


def test_standard_loader_rejects_label_artifact(tmp_path: Path, tiny_job: TrajectoryJob) -> None:
    _, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        {"schema_version": 1},
    )
    with pytest.raises(ValueError, match="observable artifact required"):
        load_observable(label_path)


def test_loader_rejects_a_checksum_corrupted_artifact(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, _ = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        {"schema_version": 1},
    )
    (observable_path / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        load_observable(observable_path)


def test_validation_accepts_a_chunk_with_absolute_episode_numbers() -> None:
    observable = _observable()
    validate_observable(
        replace(
            observable,
            global_round=observable.global_round + np.uint32(512),
            episode=observable.episode + np.uint32(16),
            block=observable.block + np.uint32(2),
            max_source_round=observable.max_source_round + 512,
        )
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("global_round", np.array([0, 1, *range(3, 65)], dtype=np.uint32)),
        ("round_in_episode", np.arange(1, 65, dtype=np.uint32) % 32),
        ("episode", np.arange(64, dtype=np.uint32) // 16),
        ("block", np.ones(64, dtype=np.uint32)),
    ],
)
def test_validation_rejects_clock_inconsistent_chronology(
    field: str, replacement: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="clock"):
        validate_observable(replace(_observable(), **{field: replacement}))


def test_publish_rejects_labels_with_a_different_round_count(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    with pytest.raises(ValueError, match="label rounds must match observable rounds"):
        publish_trajectory(
            tmp_path, tiny_job, _observable(), _labels(rounds=63), {"schema_version": 1}
        )


def test_publish_accepts_the_mapping_contract_for_metadata(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, _ = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        MappingProxyType({"schema_version": 1}),
    )
    assert load_observable(observable_path).detector_bits.shape == (64, 3)


def test_publish_rejects_raw_private_seed_metadata(tmp_path: Path, tiny_job: TrajectoryJob) -> None:
    with pytest.raises(ValueError, match="raw seed"):
        publish_trajectory(
            tmp_path,
            tiny_job,
            _observable(),
            _labels(),
            {"schema_version": 1, "private_seed": 713},
        )
    assert not (tmp_path / "data").exists()


def test_publish_rejects_a_numeric_seed_hash_metadata_value(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    with pytest.raises(ValueError, match="raw seed"):
        publish_trajectory(
            tmp_path,
            tiny_job,
            _observable(),
            _labels(),
            {"schema_version": 1, "seed_hash": 713},
        )


def test_publish_accepts_a_sha256_seed_commitment_hash(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, _ = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        {"schema_version": 1, "seed_commitment_hash": "a" * 64},
    )
    assert load_observable(observable_path).detector_bits.shape == (64, 3)


def test_publish_accepts_a_structured_sha256_seed_commitment(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, _ = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        {
            "schema_version": 1,
            "seed_commitment": {"algorithm": "sha256", "digest": "a" * 64},
        },
    )
    assert load_observable(observable_path).detector_bits.shape == (64, 3)


@pytest.mark.parametrize("lane", ["observable", "labels"])
def test_publish_rolls_back_both_lanes_after_postrename_directory_sync_failure(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    failed_parent = observable_path.parent if lane == "observable" else label_path.parent
    original_fsync = artifact_module._fsync_directory

    def fail_postrename_sync(path: Path) -> None:
        if path == failed_parent:
            raise OSError(f"simulated {lane} post-rename sync failure")
        original_fsync(path)

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_postrename_sync)
    with pytest.raises(OSError, match="post-rename sync failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert not observable_path.exists()
    assert not label_path.exists()


def test_publish_failure_preserves_a_concurrently_created_complete_pair(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    original_publish = artifact_module._publish_directory

    def simulate_concurrent_pair(staging: Path, target: Path) -> object:
        if target == observable_path:
            concurrent_label_staging = next(label_path.parent.glob(".*.staging-*"))
            shutil.copytree(staging, observable_path)
            shutil.copytree(concurrent_label_staging, label_path)
        if target == label_path:
            raise OSError("simulated concurrent label failure")
        return original_publish(staging, target)

    monkeypatch.setattr(artifact_module, "_publish_directory", simulate_concurrent_pair)
    with pytest.raises(OSError, match="concurrent label failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert verify_artifact(observable_path)
    assert verify_artifact(label_path)


def test_publish_never_overwrites_a_rival_created_at_the_no_replace_seam(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )

    def create_rival_after_check(staging: Path, target: Path) -> None:
        if target == observable_path:
            target.mkdir()
            (target / "rival").write_text("do not replace", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(
        artifact_module, "_rename_no_replace", create_rival_after_check, raising=False
    )
    with pytest.raises(ValueError, match="incomplete artifact"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert (observable_path / "rival").read_text(encoding="utf-8") == "do not replace"


def test_cleanup_continues_after_a_concurrent_label_corruption(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    original_fsync = artifact_module._fsync_directory

    def fail_after_corrupting_label(path: Path) -> None:
        if path == label_path.parent:
            (label_path / "arrays.npz").write_bytes(b"concurrent corruption")
            raise OSError("simulated post-label failure")
        original_fsync(path)

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_after_corrupting_label)
    with pytest.raises(OSError, match="post-label failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert not observable_path.exists()
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        verify_artifact(label_path)


def test_cleanup_preserves_a_rival_replacing_an_owned_lane_after_verification(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    displaced_owned_path = label_path.with_name("owned-label-displaced")
    original_fsync = artifact_module._fsync_directory
    original_verify = artifact_module.verify_artifact

    def fail_post_label_sync(path: Path) -> None:
        if path == label_path.parent:
            raise OSError("simulated post-label failure")
        original_fsync(path)

    def replace_label_after_cleanup_verification(path: Path) -> str:
        artifact_hash = original_verify(path)
        if path == label_path:
            path.rename(displaced_owned_path)
            path.mkdir()
            (path / "rival").write_text("preserve me", encoding="utf-8")
        return artifact_hash

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_post_label_sync)
    monkeypatch.setattr(
        artifact_module, "verify_artifact", replace_label_after_cleanup_verification
    )
    with pytest.raises(OSError, match="post-label failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert not observable_path.exists()
    assert (label_path / "rival").read_text(encoding="utf-8") == "preserve me"


def test_cleanup_continues_after_a_label_removal_failure(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    original_fsync = artifact_module._fsync_directory
    original_rmtree = artifact_module.shutil.rmtree

    def fail_post_label_sync(path: Path) -> None:
        if path == label_path.parent:
            raise OSError("simulated post-label failure")
        original_fsync(path)

    def fail_label_removal(path: Path, *args: object, **kwargs: object) -> None:
        if path == label_path or (
            path.parent == label_path.parent and path.name.startswith(".0.cleanup-")
        ):
            raise OSError("simulated label removal failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_post_label_sync)
    monkeypatch.setattr(artifact_module.shutil, "rmtree", fail_label_removal)
    with pytest.raises(OSError, match="post-label failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert not observable_path.exists()
    assert label_path.exists()


def test_publish_removes_first_lane_when_label_publication_fails(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    observable_path = (
        tmp_path / "data" / "observable" / tiny_job.split / tiny_job.condition_id / "0"
    )
    label_path = tmp_path / "data" / "labels" / tiny_job.split / tiny_job.condition_id / "0"
    original_rename = artifact_module._rename_no_replace

    def fail_label_publish(source: Path, target: Path) -> None:
        if target == label_path:
            raise OSError("simulated label publication failure")
        original_rename(source, target)

    monkeypatch.setattr(artifact_module, "_rename_no_replace", fail_label_publish)
    with pytest.raises(OSError, match="label publication failure"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert not observable_path.exists()
    assert not label_path.exists()


def test_publish_rejects_an_existing_one_lane_artifact_pair(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    _, label_path = publish_trajectory(
        tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1}
    )
    shutil.rmtree(label_path)
    with pytest.raises(ValueError, match="incomplete artifact pair"):
        publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})


def test_publish_resumes_only_an_identical_artifact_pair(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    first = publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    first_hashes = tuple(verify_artifact(path) for path in first)
    second = publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert second == first
    assert tuple(verify_artifact(path) for path in second) == first_hashes


def test_publish_never_overwrites_conflicting_complete_artifact(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable = _observable()
    labels = _labels()
    publish_trajectory(tmp_path, tiny_job, observable, labels, {"schema_version": 1})
    changed = replace(observable, detector_bits=~observable.detector_bits)
    with pytest.raises(FileExistsError, match="artifact conflict"):
        publish_trajectory(tmp_path, tiny_job, changed, labels, {"schema_version": 1})
