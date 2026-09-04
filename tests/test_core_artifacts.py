import errno
import json
import shutil
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import causaldem_qec.artifacts as artifact_module
from causaldem_qec.artifacts import (
    load_labels,
    load_observable,
    load_sealed_seed,
    publish_trajectory,
    verify_artifact,
    write_sealed_commitment,
)
from causaldem_qec.cli import _freeze_sealed
from causaldem_qec.core import (
    CircuitSpec,
    ExecutionOptions,
    LabelTrajectory,
    ManifestProvenance,
    ObservableTrajectory,
    RunManifest,
    TrajectoryJob,
    derive_seed,
    deserialize_manifest_provenance,
    expand_jobs,
    load_spec,
    serialize_manifest_provenance,
    source_cutoff,
    target_interval,
    validate_observable,
)

CONFIG = Path("configs/poc.json")
PILOT_CONFIG = Path("configs/poc_pilot.json")
CHECKPOINT_CONFIG_HASH = "a" * 64
CHECKPOINT_IDENTITY = "owner/causaldem-pilot-checkpoint"
CHECKPOINT_VERSION = f"{CHECKPOINT_IDENTITY}@7"
CHECKPOINT_PROVENANCE = ManifestProvenance(
    source_commit="256488b",
    execution_backend="kaggle",
    generation_law_version="standard_monolithic_v1",
    checkpoint_identity=CHECKPOINT_IDENTITY,
)
CHECKPOINT_SEALED_COMMITMENT = {"algorithm": "sha256", "digest": "b" * 64}
LEGACY_CHECKPOINT_CONFIG_HASH = "bcd3d21f1a013da6767b935bc6d452f927d13d2ec4ef0fc3326145a55a01d09c"


def test_execution_options_are_frozen_and_normalize_the_backend() -> None:
    options = ExecutionOptions(
        execution_backend=" KAGGLE ",
        job_limit=1,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_version=CHECKPOINT_VERSION,
        generation_mode="bounded",
        generation_chunk_rounds=256,
    )

    assert options.execution_backend == "kaggle"
    with pytest.raises(FrozenInstanceError):
        options.job_limit = 2  # type: ignore[misc]


@pytest.mark.parametrize("backend", ["", "remote", "kaggle-gpu"])
def test_execution_options_reject_unknown_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="execution backend"):
        ExecutionOptions(execution_backend=backend)


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_limit": 0},
        {"checkpoint_identity": " "},
        {"checkpoint_version": " "},
        {"generation_mode": "streaming"},
        {"generation_chunk_rounds": 256},
        {"generation_mode": "bounded", "generation_chunk_rounds": 0},
    ],
)
def test_execution_options_reject_invalid_limits_and_bounded_settings(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ExecutionOptions(**overrides)  # type: ignore[arg-type]


def test_manifest_provenance_round_trips_without_raw_seed_fields() -> None:
    provenance = ManifestProvenance(
        source_commit="1440540b725772582122235ccf845f4f62347ad6",
        execution_backend="KAGGLE",
        generation_law_version="standard_monolithic_v1",
        checkpoint_identity=CHECKPOINT_IDENTITY,
        generation_mode="bounded",
        generation_chunk_rounds=256,
    )

    encoded = serialize_manifest_provenance(provenance)
    decoded = deserialize_manifest_provenance(json.loads(json.dumps(encoded)))

    assert decoded == provenance
    assert encoded == {
        "source_commit": "1440540b725772582122235ccf845f4f62347ad6",
        "execution_backend": "kaggle",
        "generation_law_version": "standard_monolithic_v1",
        "checkpoint_identity": CHECKPOINT_IDENTITY,
        "generation_mode": "bounded",
        "generation_chunk_rounds": 256,
    }
    assert "seed" not in json.dumps(encoded).lower()


def test_manifest_provenance_rejects_raw_seed_fields() -> None:
    encoded = serialize_manifest_provenance(
        ManifestProvenance(
            source_commit="1440540b725772582122235ccf845f4f62347ad6",
            execution_backend="kaggle",
            generation_law_version="standard_monolithic_v1",
            checkpoint_identity=None,
            generation_mode="standard",
            generation_chunk_rounds=None,
        )
    )

    with pytest.raises(ValueError, match="unknown manifest provenance keys"):
        deserialize_manifest_provenance({**encoded, "root_seed": 713})


def test_run_manifest_keeps_provenance_optional_for_existing_callers() -> None:
    manifest = RunManifest(0, 0, 0, MappingProxyType({}), (), "manifest-hash")

    assert manifest.provenance is None


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


def test_pilot_config_is_an_additive_reduced_profile() -> None:
    production = load_spec(CONFIG)
    pilot = load_spec(PILOT_CONFIG)
    assert production.dataset_profile == "production"
    assert len(expand_jobs(production, include_sealed=True)) == 1920
    assert (production.burn_in_rounds, production.scored_rounds) == (4096, 65536)
    assert pilot.dataset_profile == "pilot"
    assert (pilot.burn_in_rounds, pilot.scored_rounds) == (4096, 8192)


def test_pilot_covers_every_condition_once_in_its_declared_partition() -> None:
    pilot = load_spec(PILOT_CONFIG)
    jobs = expand_jobs(pilot, include_sealed=True)
    assert len(jobs) == 88
    assert {job.condition_id for job in jobs} == set(pilot.condition_ids)
    assert sum(job.split == "train" for job in jobs) == 20
    assert sum(job.split == "validation" for job in jobs) == 10
    assert sum(job.split == "id_test" for job in jobs) == 10
    assert sum(job.split == "development" for job in jobs) == 24
    assert sum(job.split == "sealed_test" for job in jobs) == 24
    assert len(expand_jobs(pilot, include_sealed=False)) == 64


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dataset_profile", "not-a-profile", "dataset_profile"),
        ("rounds", {"burn_in": 4096, "scored": 65536, "episode": 32, "block": 256}, "pilot"),
    ],
)
def test_pilot_profile_rejects_invalid_identity_and_geometry(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    document = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    document[field] = value
    path = tmp_path / "invalid-pilot.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_spec(path)


@pytest.mark.parametrize("partition", ["normal", "development"])
def test_pilot_profile_rejects_missing_or_duplicate_condition_partition(
    tmp_path: Path, partition: str
) -> None:
    document = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    if partition == "normal":
        document["pilot_partitions"]["normal"] = document["pilot_partitions"]["normal"][1:]
    else:
        document["pilot_partitions"]["development"].append(
            document["pilot_partitions"]["normal"][0]
        )
    path = tmp_path / "invalid-pilot.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot condition"):
        load_spec(path)


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


def _checkpoint_metadata(
    config_hash: str = CHECKPOINT_CONFIG_HASH,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_profile": "pilot",
        "resolved_config_hash": config_hash,
    }


def _checkpoint_inventory(root: Path) -> tuple[artifact_module.CheckpointPair, ...]:
    return artifact_module.inventory_checkpoint(
        root,
        expected_config_hash=CHECKPOINT_CONFIG_HASH,
        expected_provenance=CHECKPOINT_PROVENANCE,
        expected_checkpoint_version=CHECKPOINT_VERSION,
    )


def _checkpoint_export(source: Path, destination: Path) -> Path:
    return artifact_module.export_checkpoint(
        source,
        destination,
        expected_config_hash=CHECKPOINT_CONFIG_HASH,
        expected_provenance=CHECKPOINT_PROVENANCE,
        expected_checkpoint_version=CHECKPOINT_VERSION,
    )


def _write_checkpoint_manifest(
    root: Path,
    job: TrajectoryJob,
    observable_path: Path,
    label_path: Path,
    *,
    observable_hash: str | None = None,
    label_hash: str | None = None,
    pair_id: str | None = None,
) -> bytes:
    observable_metadata = json.loads(
        (observable_path / "metadata.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "dataset_profile": "pilot",
        "resolved_config_hash": CHECKPOINT_CONFIG_HASH,
        "provenance": serialize_manifest_provenance(CHECKPOINT_PROVENANCE),
        "checkpoint_input_version": CHECKPOINT_VERSION,
        "results": [
            {
                "condition_id": job.condition_id,
                "trajectory_id": job.trajectory_id,
                "completed": True,
                "observable_hash": observable_hash or verify_artifact(observable_path),
                "label_hash": label_hash or verify_artifact(label_path),
                "pair_id": pair_id or observable_metadata["pair_id"],
                "failure": None,
            }
        ],
    }
    if job.split == "sealed_test":
        manifest["sealed_commitment"] = CHECKPOINT_SEALED_COMMITMENT
    encoded = json.dumps(manifest, indent=2).encode()
    (root / "run_manifest.json").write_bytes(encoded)
    return encoded


def _add_checksum_valid_checkpoint_array(root: Path, artifact_path: Path, name: str) -> None:
    with np.load(artifact_path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays[name] = np.asarray([713], dtype=np.uint64)
    artifact_module._write_deterministic_npz(artifact_path / "arrays.npz", arrays)
    artifact_hash = artifact_module._write_sums(artifact_path)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"][0]["observable_hash"] = artifact_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_legacy_bootstrap_checkpoint(root: Path) -> tuple[Path, ...]:
    jobs = tuple(
        sorted(
            expand_jobs(load_spec(PILOT_CONFIG), include_sealed=True),
            key=lambda job: (job.condition_id, job.trajectory_id),
        )
    )
    completed_jobs = tuple(job for job in jobs if job.circuit.family == "repetition")
    assert len(jobs) == 88
    assert len(completed_jobs) == 44
    artifact_paths: list[Path] = []
    results: list[dict[str, object]] = []
    for job in completed_jobs:
        observable_path, label_path = publish_trajectory(
            root,
            job,
            _observable(),
            _labels(),
            _checkpoint_metadata(LEGACY_CHECKPOINT_CONFIG_HASH),
        )
        artifact_paths.extend((observable_path, label_path))
        metadata = json.loads((observable_path / "metadata.json").read_text(encoding="utf-8"))
        results.append(
            {
                "completed": True,
                "condition_id": job.condition_id,
                "failure": None,
                "label_hash": verify_artifact(label_path),
                "observable_hash": verify_artifact(observable_path),
                "pair_id": metadata["pair_id"],
                "trajectory_id": job.trajectory_id,
            }
        )
    commitment_path = root / "data" / "manifests" / "sealed_commitment.json"
    commitment_path.parent.mkdir(parents=True)
    commitment_path.write_text(json.dumps(CHECKPOINT_SEALED_COMMITMENT), encoding="utf-8")
    manifest = {
        "dataset_profile": "pilot",
        "expected_job_keys": [[job.condition_id, job.trajectory_id] for job in jobs],
        "generation": {
            "burn_in_rounds": 4096,
            "scored_rounds": 8192,
            "trajectories_per_condition": 64,
        },
        "resolved_config_hash": LEGACY_CHECKPOINT_CONFIG_HASH,
        "results": results,
        "schema_version": 1,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return tuple(artifact_paths)


@pytest.mark.parametrize("operation", ["inventory", "export"])
@pytest.mark.parametrize("omission", ["both", "config", "provenance"])
def test_public_checkpoint_workflow_requires_caller_supplied_identity(
    tmp_path: Path, operation: str, omission: str
) -> None:
    arguments: dict[str, object] = {
        "expected_config_hash": CHECKPOINT_CONFIG_HASH,
        "expected_provenance": CHECKPOINT_PROVENANCE,
        "expected_checkpoint_version": CHECKPOINT_VERSION,
    }
    if omission == "both":
        arguments.clear()
    elif omission == "config":
        del arguments["expected_config_hash"]
    else:
        del arguments["expected_provenance"]

    with pytest.raises(TypeError):
        if operation == "inventory":
            artifact_module.inventory_checkpoint(tmp_path, **arguments)  # type: ignore[arg-type]
        else:
            artifact_module.export_checkpoint(  # type: ignore[arg-type]
                tmp_path,
                tmp_path / "export",
                **arguments,
            )


def test_legacy_bootstrap_upgrade_validates_all_44_pairs_before_workflow(
    tmp_path: Path,
) -> None:
    artifact_paths = _write_legacy_bootstrap_checkpoint(tmp_path)
    before_hashes = tuple(verify_artifact(path) for path in artifact_paths)

    manifest_path = artifact_module.upgrade_legacy_kaggle_bootstrap(
        tmp_path,
        expected_config_hash=LEGACY_CHECKPOINT_CONFIG_HASH,
        expected_provenance=CHECKPOINT_PROVENANCE,
        expected_checkpoint_version=CHECKPOINT_VERSION,
    )

    upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert upgraded["provenance"] == serialize_manifest_provenance(CHECKPOINT_PROVENANCE)
    assert upgraded["checkpoint_input_version"] == CHECKPOINT_VERSION
    assert upgraded["sealed_commitment"] == CHECKPOINT_SEALED_COMMITMENT
    assert tuple(verify_artifact(path) for path in artifact_paths) == before_hashes
    inventory = artifact_module.inventory_checkpoint(
        tmp_path,
        expected_config_hash=LEGACY_CHECKPOINT_CONFIG_HASH,
        expected_provenance=CHECKPOINT_PROVENANCE,
        expected_checkpoint_version=CHECKPOINT_VERSION,
    )
    assert len(inventory) == 44
    export = tmp_path / "export"
    assert (
        artifact_module.export_checkpoint(
            tmp_path,
            export,
            expected_config_hash=LEGACY_CHECKPOINT_CONFIG_HASH,
            expected_provenance=CHECKPOINT_PROVENANCE,
            expected_checkpoint_version=CHECKPOINT_VERSION,
        )
        == export
    )


def test_legacy_bootstrap_upgrade_refuses_an_arbitrary_checkpoint(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(LEGACY_CHECKPOINT_CONFIG_HASH),
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["provenance"]
    manifest["resolved_config_hash"] = LEGACY_CHECKPOINT_CONFIG_HASH
    original = json.dumps(manifest).encode()
    manifest_path.write_bytes(original)

    with pytest.raises(artifact_module.ArtifactConflict, match="legacy bootstrap identity"):
        artifact_module.upgrade_legacy_kaggle_bootstrap(
            tmp_path,
            expected_config_hash=LEGACY_CHECKPOINT_CONFIG_HASH,
            expected_provenance=CHECKPOINT_PROVENANCE,
            expected_checkpoint_version=CHECKPOINT_VERSION,
        )
    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize("failure", ["config", "raw_seed", "corrupt_pair"])
def test_legacy_bootstrap_upgrade_rejects_invalid_input_without_rewriting_manifest(
    tmp_path: Path, failure: str
) -> None:
    artifact_paths = _write_legacy_bootstrap_checkpoint(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    expected_config_hash = LEGACY_CHECKPOINT_CONFIG_HASH
    if failure == "config":
        expected_config_hash = "c" * 64
    elif failure == "raw_seed":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["root_seed"] = 713
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (artifact_paths[0] / "arrays.npz").write_bytes(b"corrupt")
    original = manifest_path.read_bytes()

    with pytest.raises(artifact_module.ArtifactConflict):
        artifact_module.upgrade_legacy_kaggle_bootstrap(
            tmp_path,
            expected_config_hash=expected_config_hash,
            expected_provenance=CHECKPOINT_PROVENANCE,
            expected_checkpoint_version=CHECKPOINT_VERSION,
        )
    assert manifest_path.read_bytes() == original


def test_checkpoint_inventory_includes_only_manifest_verified_complete_pairs(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    staging = observable_path.parent / ".1.staging-interrupted"
    shutil.copytree(observable_path, staging)
    incomplete = observable_path.parent / "1"
    shutil.copytree(observable_path, incomplete)
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "sealed_private.json").write_text('{"root_seed": 99887766}', encoding="utf-8")

    inventory = _checkpoint_inventory(tmp_path)

    assert len(inventory) == 1
    assert inventory[0].relative_path == Path(
        "data", "observable", tiny_job.split, tiny_job.condition_id, "0"
    )
    assert inventory[0].observable_hash == verify_artifact(observable_path)
    assert inventory[0].label_hash == verify_artifact(label_path)


@pytest.mark.parametrize("array_name", ["root_seed", "unexpected_array"])
def test_checkpoint_inventory_rejects_checksum_valid_noncontract_arrays(
    tmp_path: Path, tiny_job: TrajectoryJob, array_name: str
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    _add_checksum_valid_checkpoint_array(tmp_path, observable_path, array_name)

    with pytest.raises(artifact_module.ArtifactConflict, match="schema"):
        _checkpoint_inventory(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_profile", "production"),
        ("resolved_config_hash", "c" * 64),
        ("execution_backend", "local"),
        ("source_commit", "different-commit"),
        ("generation_law_version", "bounded_surface_v1"),
        ("checkpoint_identity", "other-checkpoint:9"),
    ],
)
def test_checkpoint_inventory_rejects_manifest_identity_mismatches(
    tmp_path: Path, tiny_job: TrajectoryJob, field: str, value: str
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path, tiny_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field in manifest["provenance"]:
        manifest["provenance"][field] = value
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(artifact_module.ArtifactConflict, match="identity"):
        artifact_module.inventory_checkpoint(
            tmp_path,
            expected_config_hash=CHECKPOINT_CONFIG_HASH,
            expected_provenance=CHECKPOINT_PROVENANCE,
            expected_checkpoint_version=CHECKPOINT_VERSION,
        )


def test_checkpoint_inventory_allows_newer_version_only_for_same_stable_dataset(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path, tiny_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)

    assert artifact_module.inventory_checkpoint(
        tmp_path,
        expected_config_hash=CHECKPOINT_CONFIG_HASH,
        expected_provenance=CHECKPOINT_PROVENANCE,
        expected_checkpoint_version=f"{CHECKPOINT_IDENTITY}@8",
    )
    with pytest.raises(artifact_module.ArtifactConflict, match="checkpoint.*identity"):
        artifact_module.inventory_checkpoint(
            tmp_path,
            expected_config_hash=CHECKPOINT_CONFIG_HASH,
            expected_provenance=replace(
                CHECKPOINT_PROVENANCE,
                checkpoint_identity="other-owner/other-checkpoint",
            ),
            expected_checkpoint_version="other-owner/other-checkpoint@8",
        )
    with pytest.raises(artifact_module.ArtifactConflict, match="checkpoint.*version"):
        artifact_module.inventory_checkpoint(
            tmp_path,
            expected_config_hash=CHECKPOINT_CONFIG_HASH,
            expected_provenance=CHECKPOINT_PROVENANCE,
            expected_checkpoint_version=f"{CHECKPOINT_IDENTITY}@6",
        )


@pytest.mark.parametrize(
    "failure", ["missing_manifest_commitment", "missing_commitment_artifact", "source_mismatch"]
)
def test_checkpoint_inventory_rejects_invalid_sealed_commitment_constraints(
    tmp_path: Path, tiny_job: TrajectoryJob, failure: str
) -> None:
    sealed_job = replace(tiny_job, split="sealed_test")
    observable_path, label_path = publish_trajectory(
        tmp_path, sealed_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(tmp_path, sealed_job, observable_path, label_path)
    commitment_path = tmp_path / "data" / "manifests" / "sealed_commitment.json"
    commitment_path.parent.mkdir(parents=True)
    commitment_path.write_text(json.dumps(CHECKPOINT_SEALED_COMMITMENT), encoding="utf-8")
    if failure == "missing_manifest_commitment":
        manifest_path = tmp_path / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["sealed_commitment"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif failure == "missing_commitment_artifact":
        commitment_path.unlink()
    else:
        commitment_path.write_text(
            json.dumps({"algorithm": "sha256", "digest": "c" * 64}), encoding="utf-8"
        )

    with pytest.raises(artifact_module.ArtifactConflict, match="sealed commitment"):
        _checkpoint_inventory(tmp_path)


@pytest.mark.parametrize("field", ["observable_hash", "label_hash", "pair_id"])
def test_checkpoint_inventory_rejects_manifest_pair_conflicts(
    tmp_path: Path, tiny_job: TrajectoryJob, field: str
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    overrides = {field: "0" * 64}
    _write_checkpoint_manifest(
        tmp_path,
        tiny_job,
        observable_path,
        label_path,
        **overrides,
    )

    with pytest.raises(artifact_module.ArtifactConflict, match="identity mismatch"):
        _checkpoint_inventory(tmp_path)


def test_checkpoint_inventory_rejects_a_manifest_completed_single_lane(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    shutil.rmtree(label_path)

    with pytest.raises(artifact_module.ArtifactConflict, match="incomplete checkpoint"):
        _checkpoint_inventory(tmp_path)


def test_checkpoint_inventory_rejects_raw_seed_manifest_metadata(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(tmp_path, tiny_job, observable_path, label_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sealed_seed"] = 99887766
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(artifact_module.ArtifactConflict, match="raw seed"):
        _checkpoint_inventory(tmp_path)


def test_export_checkpoint_preserves_manifest_and_copies_only_verified_files(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    observable_path, label_path = publish_trajectory(
        source,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    manifest_bytes = _write_checkpoint_manifest(source, tiny_job, observable_path, label_path)
    (observable_path / "raw_seed.txt").write_text("99887766", encoding="utf-8")
    secrets = source / "secrets"
    secrets.mkdir()
    (secrets / "sealed_private.json").write_text('{"root_seed": 99887766}', encoding="utf-8")
    shutil.copytree(observable_path, observable_path.parent / ".1.staging-interrupted")

    assert _checkpoint_export(source, export) == export

    assert (export / "run_manifest.json").read_bytes() == manifest_bytes
    assert _checkpoint_inventory(export) == _checkpoint_inventory(source)
    relative_files = {
        path.relative_to(export).as_posix() for path in export.rglob("*") if path.is_file()
    }
    assert relative_files == {
        "run_manifest.json",
        *{
            str(relative / name)
            for relative in (
                Path("data", "observable", tiny_job.split, tiny_job.condition_id, "0"),
                Path("data", "labels", tiny_job.split, tiny_job.condition_id, "0"),
            )
            for name in ("arrays.npz", "metadata.json", "SHA256SUMS")
        },
    }


def test_export_checkpoint_carries_only_public_sealed_commitment_for_resume(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    private = tmp_path / "private" / "sealed.json"
    private.parent.mkdir()
    private.write_text('{"root_seed":99887766}', encoding="utf-8")
    sealed_job = replace(tiny_job, split="sealed_test")
    observable_path, label_path = publish_trajectory(
        source, sealed_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(source, sealed_job, observable_path, label_path)
    commitment_path = source / "data" / "manifests" / "sealed_commitment.json"
    digest = write_sealed_commitment(private, commitment_path)
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sealed_commitment"] = {"algorithm": "sha256", "digest": digest}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _checkpoint_export(source, export)

    exported_commitment = export / "data" / "manifests" / "sealed_commitment.json"
    assert exported_commitment.read_bytes() == commitment_path.read_bytes()
    assert load_sealed_seed(private, exported_commitment, purpose="sealed_evaluation") == 99887766
    assert not any(path.name == private.name for path in export.rglob("*"))


def test_export_checkpoint_never_overwrites_a_conflicting_destination(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    observable_path, label_path = publish_trajectory(
        source,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(source, tiny_job, observable_path, label_path)
    export.mkdir()
    marker = export / "preserve.txt"
    marker.write_text("existing export", encoding="utf-8")

    with pytest.raises(artifact_module.ArtifactConflict, match="checkpoint export conflict"):
        _checkpoint_export(source, export)
    assert marker.read_text(encoding="utf-8") == "existing export"


def test_export_checkpoint_rejects_a_symlink_destination(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    source = tmp_path / "source"
    real_export = tmp_path / "real-export"
    linked_export = tmp_path / "linked-export"
    observable_path, label_path = publish_trajectory(
        source, tiny_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(source, tiny_job, observable_path, label_path)
    _checkpoint_export(source, real_export)
    linked_export.symlink_to(real_export, target_is_directory=True)

    with pytest.raises(artifact_module.ArtifactConflict, match="checkpoint export conflict"):
        _checkpoint_export(source, linked_export)


@pytest.mark.parametrize("linked_file", ["manifest", "artifact"])
def test_export_checkpoint_rejects_symlink_files_in_an_existing_destination(
    tmp_path: Path, tiny_job: TrajectoryJob, linked_file: str
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    observable_path, label_path = publish_trajectory(
        source, tiny_job, _observable(), _labels(), _checkpoint_metadata()
    )
    _write_checkpoint_manifest(source, tiny_job, observable_path, label_path)
    _checkpoint_export(source, export)
    relative = (
        Path("run_manifest.json")
        if linked_file == "manifest"
        else Path("data", "observable", tiny_job.split, tiny_job.condition_id, "0", "arrays.npz")
    )
    linked_path = export / relative
    linked_path.unlink()
    linked_path.symlink_to(source / relative)

    with pytest.raises(artifact_module.ArtifactConflict, match="checkpoint export conflict"):
        _checkpoint_export(source, export)


def test_export_checkpoint_recovers_after_postrename_sync_failure(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    observable_path, label_path = publish_trajectory(
        source,
        tiny_job,
        _observable(),
        _labels(),
        _checkpoint_metadata(),
    )
    _write_checkpoint_manifest(source, tiny_job, observable_path, label_path)
    original_fsync = artifact_module._fsync_directory

    def fail_export_parent_sync(path: Path) -> None:
        if path == export.parent:
            raise OSError("simulated checkpoint post-rename sync failure")
        original_fsync(path)

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_export_parent_sync)
    with pytest.raises(OSError, match="post-rename sync failure"):
        _checkpoint_export(source, export)
    assert _checkpoint_inventory(export)

    monkeypatch.setattr(artifact_module, "_fsync_directory", original_fsync)
    assert _checkpoint_export(source, export) == export


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


def test_publish_uses_job_private_staging_outside_the_artifact_tree(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    """Catch bounded staging paths being mistaken for a completed observable/label artifact."""
    staging_root = tmp_path / ".staging" / tiny_job.condition_id / str(tiny_job.trajectory_id) / "0"

    observable_path, label_path = publish_trajectory(
        tmp_path,
        tiny_job,
        _observable(),
        _labels(),
        {"schema_version": 1},
        staging_root=staging_root,
    )

    assert verify_artifact(observable_path)
    assert verify_artifact(label_path)
    assert not staging_root.exists()


def test_npz_writer_streams_members_without_a_whole_array_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch publication buffering an entire array in BytesIO before ZIP output."""
    path = tmp_path / "arrays.npz"
    monkeypatch.setattr(
        artifact_module.zipfile.ZipFile,
        "writestr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("whole-array buffer allocated")),
    )
    artifact_module._write_deterministic_npz(path, {"values": np.arange(16, dtype=np.float64)})

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], np.arange(16, dtype=np.float64))


def test_publish_never_overwrites_conflicting_complete_artifact(
    tmp_path: Path, tiny_job: TrajectoryJob
) -> None:
    observable = _observable()
    labels = _labels()
    publish_trajectory(tmp_path, tiny_job, observable, labels, {"schema_version": 1})
    changed = replace(observable, detector_bits=~observable.detector_bits)
    with pytest.raises(FileExistsError, match="artifact conflict"):
        publish_trajectory(tmp_path, tiny_job, changed, labels, {"schema_version": 1})


def test_development_command_cannot_open_private_sealed_manifest(tmp_path: Path) -> None:
    private = tmp_path / "sealed_private.json"
    commitment = tmp_path / "sealed_commitment.json"
    private.write_text('{"root_seed": 99887766}', encoding="utf-8")
    write_sealed_commitment(private, commitment)
    with pytest.raises(PermissionError, match="sealed evaluation only"):
        load_sealed_seed(private, commitment, purpose="development")
    assert load_sealed_seed(private, commitment, purpose="sealed_evaluation") == 99887766


def test_publish_uses_atomic_replace_when_renameat2_is_unsupported(
    tmp_path: Path, tiny_job: TrajectoryJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*args: object) -> None:
        raise OSError(errno.EINVAL, "renameat2 unsupported")

    monkeypatch.setattr(artifact_module, "_rename_with_flags", unsupported)
    paths = publish_trajectory(tmp_path, tiny_job, _observable(), _labels(), {"schema_version": 1})
    assert tuple(verify_artifact(path) for path in paths)


def test_freeze_refuses_an_existing_commitment_without_creating_private_seed(
    tmp_path: Path,
) -> None:
    spec = load_spec(CONFIG)
    private = tmp_path / "private.json"
    commitment = tmp_path / "data" / "manifests" / "sealed_commitment.json"
    commitment.parent.mkdir(parents=True)
    commitment.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="sealed commitment exists"):
        _freeze_sealed(spec, private, tmp_path)
    assert not private.exists()


def test_freeze_creates_private_seed_exclusively_with_mode_0600(tmp_path: Path) -> None:
    spec = load_spec(CONFIG)
    private = tmp_path / "private.json"
    digest = _freeze_sealed(spec, private, tmp_path)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _freeze_sealed(spec, private, tmp_path)
    assert (tmp_path / "data" / "manifests" / "sealed_commitment.json").read_bytes()
    assert digest
