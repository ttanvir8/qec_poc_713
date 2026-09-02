from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from causaldem_qec.core import (
    LabelTrajectory,
    ManifestProvenance,
    ObservableTrajectory,
    TrajectoryJob,
    deserialize_manifest_provenance,
    validate_labels,
    validate_observable,
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


class ArtifactConflict(FileExistsError, ValueError):
    """A complete artifact pair already exists but cannot be safely reused."""


@dataclass(frozen=True, slots=True)
class CheckpointPair:
    """A manifest-committed artifact pair safe to include in a checkpoint."""

    relative_path: Path
    label_relative_path: Path
    observable_hash: str
    label_hash: str
    pair_id: str


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Hash canonical public metadata without exposing a seed value."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    if os.name == "posix":
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            data = io.BytesIO()
            np.lib.format.write_array(data, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data.getvalue())
    _fsync_file(path)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value))
    _fsync_file(path)


def _write_sums(directory: Path) -> str:
    entries = []
    for name in ("arrays.npz", "metadata.json"):
        entries.append(f"{_sha256(directory / name)}  {name}\n")
    sums = "".join(entries)
    (directory / "SHA256SUMS").write_text(sums, encoding="ascii")
    _fsync_file(directory / "SHA256SUMS")
    _fsync_directory(directory)
    return hashlib.sha256(sums.encode("ascii")).hexdigest()


def verify_artifact(path: Path) -> str:
    """Return an artifact's checksum or reject missing, incomplete, and corrupt data."""
    sums_path = path / "SHA256SUMS"
    try:
        sums = sums_path.read_text(encoding="ascii")
    except OSError as error:
        raise ValueError(f"incomplete artifact: {path}") from error
    expected = []
    try:
        for name in ("arrays.npz", "metadata.json"):
            expected.append(f"{_sha256(path / name)}  {name}\n")
    except OSError as error:
        raise ValueError(f"incomplete artifact: {path}") from error
    if sums != "".join(expected):
        raise ValueError(f"artifact checksum mismatch: {path}")
    return hashlib.sha256(sums.encode("ascii")).hexdigest()


def _rename_with_flags(source: Path, target: Path, flags: int) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is required for safe publication") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(target),
            flags,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _rename_no_replace(source: Path, target: Path) -> None:
    _rename_with_flags(source, target, _RENAME_NOREPLACE)


def _rename_exchange(source: Path, target: Path) -> None:
    _rename_with_flags(source, target, _RENAME_EXCHANGE)


def _directory_identity(path: Path) -> tuple[int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino


def _publish_directory(staging: Path, target: Path) -> tuple[str, tuple[int, int] | None]:
    artifact_hash = _write_sums(staging)
    staging_identity = _directory_identity(staging)
    try:
        _rename_no_replace(staging, target)
    except FileExistsError:
        if verify_artifact(target) == artifact_hash:
            shutil.rmtree(staging)
            return artifact_hash, None
        raise ArtifactConflict(f"artifact conflict: {target}")
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
        lock_path = target.with_name(f".{target.name}.publish.lock")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as lock_error:
            raise ArtifactConflict(f"artifact publication is busy: {target}") from lock_error
        try:
            os.close(descriptor)
            if target.exists():
                if verify_artifact(target) == artifact_hash:
                    shutil.rmtree(staging)
                    return artifact_hash, None
                raise ArtifactConflict(f"artifact conflict: {target}")
            os.replace(staging, target)
            _fsync_directory(target.parent)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    return artifact_hash, staging_identity


def _job_metadata(job: TrajectoryJob) -> dict[str, object]:
    return {
        "condition_id": job.condition_id,
        "trajectory_id": job.trajectory_id,
        "split": job.split,
        "circuit_id": job.circuit.circuit_id,
        "circuit_family": job.circuit.family,
        "circuit_distance": job.circuit.distance,
        "dynamics_id": job.dynamics_id,
    }


def _pair_id(job: TrajectoryJob, metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json({"job": _job_metadata(job), "metadata": metadata})
    ).hexdigest()


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_sha256_commitment(value: object) -> bool:
    if _is_sha256_digest(value):
        return True
    if not isinstance(value, Mapping):
        return False
    return value.get("algorithm") == "sha256" and _is_sha256_digest(value.get("digest"))


def _reject_raw_seed_metadata(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "seed" in normalized:
                if "hash" not in normalized and "commitment" not in normalized:
                    raise ValueError("public metadata must not contain a raw seed")
                if not _is_sha256_commitment(item):
                    raise ValueError("public metadata must not contain a raw seed")
            _reject_raw_seed_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_raw_seed_metadata(item)


def _staging_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))


def _observable_arrays(trajectory: ObservableTrajectory) -> dict[str, np.ndarray]:
    return {
        "block": trajectory.block,
        "circuit_phase": trajectory.circuit_phase,
        "detector_bits_packed": np.packbits(trajectory.detector_bits, axis=1, bitorder="little"),
        "detector_role": trajectory.detector_role,
        "detector_shape": np.asarray(trajectory.detector_bits.shape, dtype=np.uint64),
        "detector_valid_packed": np.packbits(trajectory.detector_valid, axis=1, bitorder="little"),
        "episode": trajectory.episode,
        "global_round": trajectory.global_round,
        "logical_observable": trajectory.logical_observable,
        "max_source_round": trajectory.max_source_round,
        "round_in_episode": trajectory.round_in_episode,
    }


def _label_arrays(trajectory: LabelTrajectory) -> dict[str, np.ndarray]:
    return {
        "class_probability": trajectory.class_probability,
        "component_probability": trajectory.component_probability,
        "future_block_probability": trajectory.future_block_probability,
        "latent_factor": trajectory.latent_factor,
    }


def _write_lane(
    staging: Path,
    *,
    artifact_kind: Literal["observable", "labels"],
    job: TrajectoryJob,
    metadata: Mapping[str, object],
    pair_id: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    _write_deterministic_npz(staging / "arrays.npz", arrays)
    _write_json(
        staging / "metadata.json",
        {
            "artifact_kind": artifact_kind,
            "job": _job_metadata(job),
            "metadata": dict(metadata),
            "pair_id": pair_id,
        },
    )


def _existing_artifact_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return verify_artifact(path)


def _preflight_pair(
    observable_path: Path,
    label_path: Path,
    observable_hash: str,
    label_hash: str,
) -> bool:
    existing_observable = _existing_artifact_hash(observable_path)
    existing_labels = _existing_artifact_hash(label_path)
    if (existing_observable is None) != (existing_labels is None):
        raise ArtifactConflict("incomplete artifact pair")
    if existing_observable is None:
        return False
    if existing_observable != observable_hash or existing_labels != label_hash:
        raise ArtifactConflict(f"artifact conflict: {observable_path}")
    return True


def _remove_owned_artifact(path: Path, artifact_hash: str, owned_identity: tuple[int, int]) -> None:
    cleanup_path: Path | None = None
    cleanup_identity: tuple[int, int] | None = None
    try:
        matches_owned_artifact = path.exists() and verify_artifact(path) == artifact_hash
        if not matches_owned_artifact:
            return
        cleanup_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.cleanup-", dir=path.parent))
        cleanup_identity = _directory_identity(cleanup_path)
        _rename_exchange(path, cleanup_path)
        if _directory_identity(cleanup_path) != owned_identity:
            if _directory_identity(path) == cleanup_identity:
                _rename_exchange(path, cleanup_path)
            return
        try:
            shutil.rmtree(cleanup_path)
        except OSError:
            if _directory_identity(path) == cleanup_identity:
                _rename_exchange(path, cleanup_path)
            return
        _rename_no_replace(path, cleanup_path)
        if _directory_identity(cleanup_path) != cleanup_identity:
            _rename_no_replace(cleanup_path, path)
            return
        cleanup_path.rmdir()
    except (OSError, ValueError):
        return
    finally:
        if cleanup_path is not None:
            try:
                if (
                    cleanup_identity is not None
                    and _directory_identity(cleanup_path) == cleanup_identity
                ):
                    cleanup_path.rmdir()
            except OSError:
                pass
    try:
        _fsync_directory(path.parent)
    except OSError:
        pass


def publish_trajectory(
    root: Path,
    job: TrajectoryJob,
    observable: ObservableTrajectory,
    labels: LabelTrajectory,
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    """Atomically publish a verified detector-only and offline-label artifact pair."""
    validate_observable(observable)
    validate_labels(labels)
    if labels.component_probability.shape[0] != observable.detector_bits.shape[0]:
        raise ValueError("label rounds must match observable rounds")
    _reject_raw_seed_metadata(metadata)
    observable_path = (
        root / "data" / "observable" / job.split / job.condition_id / str(job.trajectory_id)
    )
    label_path = root / "data" / "labels" / job.split / job.condition_id / str(job.trajectory_id)
    pair_id = _pair_id(job, metadata)
    observable_staging = _staging_directory(observable_path)
    label_staging = _staging_directory(label_path)
    observable_identity: tuple[int, int] | None = None
    label_identity: tuple[int, int] | None = None
    try:
        _write_lane(
            observable_staging,
            artifact_kind="observable",
            job=job,
            metadata=metadata,
            pair_id=pair_id,
            arrays=_observable_arrays(observable),
        )
        _write_lane(
            label_staging,
            artifact_kind="labels",
            job=job,
            metadata=metadata,
            pair_id=pair_id,
            arrays=_label_arrays(labels),
        )
        observable_hash = _write_sums(observable_staging)
        label_hash = _write_sums(label_staging)
        if verify_artifact(observable_staging) != observable_hash:
            raise ValueError("observable staging checksum mismatch")
        if verify_artifact(label_staging) != label_hash:
            raise ValueError("label staging checksum mismatch")
        if _preflight_pair(observable_path, label_path, observable_hash, label_hash):
            shutil.rmtree(observable_staging)
            shutil.rmtree(label_staging)
            return observable_path, label_path
        _, observable_identity = _publish_directory(observable_staging, observable_path)
        _fsync_directory(observable_path.parent)
        _, label_identity = _publish_directory(label_staging, label_path)
        _fsync_directory(label_path.parent)
    except BaseException:
        if label_identity is not None:
            _remove_owned_artifact(label_path, label_hash, label_identity)
        if observable_identity is not None:
            _remove_owned_artifact(observable_path, observable_hash, observable_identity)
        if observable_staging.exists():
            shutil.rmtree(observable_staging)
        if label_staging.exists():
            shutil.rmtree(label_staging)
        raise
    return observable_path, label_path


def _read_verified_metadata(path: Path) -> Mapping[str, object]:
    verify_artifact(path)
    try:
        value: object = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact metadata: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"invalid artifact metadata: {path}")
    return value


def _array(npz: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return np.asarray(npz[name]).copy()
    except KeyError as error:
        raise ValueError(f"artifact is missing {name}") from error


def _observable_from_npz(path: Path) -> ObservableTrajectory:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {
            "block",
            "circuit_phase",
            "detector_bits_packed",
            "detector_role",
            "detector_shape",
            "detector_valid_packed",
            "episode",
            "global_round",
            "logical_observable",
            "max_source_round",
            "round_in_episode",
        }:
            raise ValueError("observable artifact array schema mismatch")
        shape = _array(data, "detector_shape")
        if shape.shape != (2,) or shape.dtype != np.uint64:
            raise ValueError("invalid detector_shape")
        rounds, detectors = (int(shape[0]), int(shape[1]))
        packed_bits = _array(data, "detector_bits_packed")
        packed_valid = _array(data, "detector_valid_packed")
        packed_width = (detectors + 7) // 8
        if packed_bits.shape != (rounds, packed_width) or packed_valid.shape != (
            rounds,
            packed_width,
        ):
            raise ValueError("packed detector arrays do not match detector_shape")
        observable = ObservableTrajectory(
            detector_bits=np.unpackbits(packed_bits, axis=1, bitorder="little")[
                :, :detectors
            ].astype(np.bool_, copy=False),
            detector_valid=np.unpackbits(packed_valid, axis=1, bitorder="little")[
                :, :detectors
            ].astype(np.bool_, copy=False),
            logical_observable=_array(data, "logical_observable"),
            global_round=_array(data, "global_round"),
            episode=_array(data, "episode"),
            round_in_episode=_array(data, "round_in_episode"),
            block=_array(data, "block"),
            detector_role=_array(data, "detector_role"),
            circuit_phase=_array(data, "circuit_phase"),
            max_source_round=_array(data, "max_source_round"),
        )
    validate_observable(observable)
    return observable


def _labels_from_npz(path: Path) -> LabelTrajectory:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {
            "class_probability",
            "component_probability",
            "future_block_probability",
            "latent_factor",
        }:
            raise ValueError("label artifact array schema mismatch")
        labels = LabelTrajectory(
            component_probability=_array(data, "component_probability"),
            latent_factor=_array(data, "latent_factor"),
            class_probability=_array(data, "class_probability"),
            future_block_probability=_array(data, "future_block_probability"),
        )
    validate_labels(labels)
    return labels


def load_observable(path: Path) -> ObservableTrajectory:
    metadata = _read_verified_metadata(path)
    if metadata.get("artifact_kind") != "observable":
        raise ValueError("observable artifact required")
    return _observable_from_npz(path / "arrays.npz")


def load_labels(path: Path, *, purpose: Literal["offline_evaluation"]) -> LabelTrajectory:
    if purpose != "offline_evaluation":
        raise ValueError("labels require explicit offline_evaluation purpose")
    metadata = _read_verified_metadata(path)
    if metadata.get("artifact_kind") != "labels":
        raise ValueError("label artifact required")
    return _labels_from_npz(path / "arrays.npz")


def trajectory_paths(root: Path, job: TrajectoryJob) -> tuple[Path, Path]:
    observable = (
        root / "data" / "observable" / job.split / job.condition_id / str(job.trajectory_id)
    )
    labels = root / "data" / "labels" / job.split / job.condition_id / str(job.trajectory_id)
    return observable, labels


def verify_trajectory_pair(
    root: Path, job: TrajectoryJob, *, resolved_config_hash: str
) -> tuple[str, str, str] | None:
    """Return hashes only for a matching, independently verified artifact pair."""
    observable_path, label_path = trajectory_paths(root, job)
    observable_exists, label_exists = observable_path.exists(), label_path.exists()
    if observable_exists != label_exists:
        raise ArtifactConflict("incomplete artifact pair")
    if not observable_exists:
        return None
    observable_hash, label_hash = verify_artifact(observable_path), verify_artifact(label_path)
    observable_metadata = _read_verified_metadata(observable_path)
    label_metadata = _read_verified_metadata(label_path)
    expected_job = _job_metadata(job)
    if (
        observable_metadata.get("artifact_kind") != "observable"
        or label_metadata.get("artifact_kind") != "labels"
        or observable_metadata.get("job") != expected_job
        or label_metadata.get("job") != expected_job
        or observable_metadata.get("pair_id") != label_metadata.get("pair_id")
        or not isinstance(observable_metadata.get("pair_id"), str)
    ):
        raise ArtifactConflict("artifact pair identity mismatch")
    observable_run = observable_metadata.get("metadata")
    label_run = label_metadata.get("metadata")
    if (
        not isinstance(observable_run, Mapping)
        or not isinstance(label_run, Mapping)
        or observable_run != label_run
        or observable_run.get("resolved_config_hash") != resolved_config_hash
    ):
        raise ArtifactConflict("artifact pair configuration mismatch")
    return observable_hash, label_hash, str(observable_metadata["pair_id"])


def _checkpoint_lane_index(
    root: Path, lane: Literal["observable", "labels"]
) -> dict[tuple[str, int], Path]:
    lane_root = root / "data" / lane
    if not lane_root.exists():
        return {}
    indexed: dict[tuple[str, int], Path] = {}
    for path in sorted(lane_root.glob("*/*/*")):
        relative = path.relative_to(lane_root)
        if (
            not path.is_dir()
            or path.is_symlink()
            or any(part.startswith(".") for part in relative.parts)
            or any(part.lower() in {"secret", "secrets", "staging"} for part in relative.parts)
        ):
            continue
        _split, condition_id, trajectory = relative.parts
        try:
            trajectory_id = int(trajectory)
        except ValueError:
            continue
        key = condition_id, trajectory_id
        if key in indexed:
            raise ArtifactConflict(
                f"duplicate checkpoint artifact pair: {condition_id}:{trajectory_id}"
            )
        indexed[key] = path
    return indexed


def _checkpoint_manifest(
    root: Path,
    *,
    expected_config_hash: str | None,
    expected_provenance: ManifestProvenance | None,
) -> Mapping[str, object]:
    try:
        value: object = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactConflict("invalid checkpoint manifest") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("results"), list):
        raise ArtifactConflict("invalid checkpoint manifest")
    try:
        _reject_raw_seed_metadata(value)
    except ValueError as error:
        raise ArtifactConflict("checkpoint manifest contains raw seed data") from error
    try:
        provenance = deserialize_manifest_provenance(value.get("provenance"))
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactConflict("checkpoint manifest identity is invalid") from error
    config_hash = value.get("resolved_config_hash")
    if (
        value.get("schema_version") != 1
        or value.get("dataset_profile") != "pilot"
        or not _is_sha256_digest(config_hash)
        or provenance.execution_backend != "kaggle"
        or provenance.checkpoint_identity is None
        or (expected_config_hash is not None and config_hash != expected_config_hash)
        or (expected_provenance is not None and provenance != expected_provenance)
    ):
        raise ArtifactConflict("checkpoint manifest identity mismatch")
    return value


def _verified_checkpoint_pair(
    root: Path,
    observable_path: Path,
    label_path: Path,
    result: Mapping[str, object],
    resolved_config_hash: str,
    dataset_profile: str,
) -> CheckpointPair:
    try:
        observable_hash = verify_artifact(observable_path)
        label_hash = verify_artifact(label_path)
        observable_metadata = _read_verified_metadata(observable_path)
        label_metadata = _read_verified_metadata(label_path)
        _reject_raw_seed_metadata(observable_metadata)
        _reject_raw_seed_metadata(label_metadata)
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactConflict("checkpoint artifact verification failed") from error
    try:
        load_observable(observable_path)
        load_labels(label_path, purpose="offline_evaluation")
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactConflict("checkpoint artifact schema validation failed") from error
    observable_relative = observable_path.relative_to(root)
    label_relative = label_path.relative_to(root)
    observable_lane_relative = observable_relative.relative_to(Path("data", "observable"))
    label_lane_relative = label_relative.relative_to(Path("data", "labels"))
    expected_job = observable_metadata.get("job")
    observable_run = observable_metadata.get("metadata")
    label_run = label_metadata.get("metadata")
    pair_id = observable_metadata.get("pair_id")
    if (
        observable_lane_relative != label_lane_relative
        or observable_metadata.get("artifact_kind") != "observable"
        or label_metadata.get("artifact_kind") != "labels"
        or not isinstance(expected_job, Mapping)
        or label_metadata.get("job") != expected_job
        or expected_job.get("split") != observable_lane_relative.parts[0]
        or expected_job.get("condition_id") != result.get("condition_id")
        or expected_job.get("trajectory_id") != result.get("trajectory_id")
        or not isinstance(pair_id, str)
        or label_metadata.get("pair_id") != pair_id
        or not isinstance(observable_run, Mapping)
        or observable_run != label_run
        or observable_run.get("resolved_config_hash") != resolved_config_hash
        or observable_run.get("dataset_profile") != dataset_profile
        or result.get("observable_hash") != observable_hash
        or result.get("label_hash") != label_hash
        or result.get("pair_id") != pair_id
    ):
        raise ArtifactConflict("checkpoint artifact pair identity mismatch")
    return CheckpointPair(
        relative_path=observable_relative,
        label_relative_path=label_relative,
        observable_hash=observable_hash,
        label_hash=label_hash,
        pair_id=pair_id,
    )


def inventory_checkpoint(
    root: Path,
    *,
    expected_config_hash: str | None = None,
    expected_provenance: ManifestProvenance | None = None,
) -> tuple[CheckpointPair, ...]:
    """Return only manifest-committed, independently verified artifact pairs."""
    manifest = _checkpoint_manifest(
        root,
        expected_config_hash=expected_config_hash,
        expected_provenance=expected_provenance,
    )
    resolved_config_hash = manifest.get("resolved_config_hash")
    if not isinstance(resolved_config_hash, str) or not resolved_config_hash:
        raise ArtifactConflict("invalid checkpoint manifest configuration")
    observable_paths = _checkpoint_lane_index(root, "observable")
    label_paths = _checkpoint_lane_index(root, "labels")
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    results = manifest["results"]
    assert isinstance(results, list)
    for value in results:
        if not isinstance(value, Mapping):
            raise ArtifactConflict("invalid checkpoint manifest result")
        if value.get("completed") is not True:
            continue
        condition_id, trajectory_id = value.get("condition_id"), value.get("trajectory_id")
        if (
            not isinstance(condition_id, str)
            or not condition_id
            or isinstance(trajectory_id, bool)
            or not isinstance(trajectory_id, int)
        ):
            raise ArtifactConflict("invalid completed checkpoint result")
        key = condition_id, trajectory_id
        if key in completed:
            raise ArtifactConflict(
                f"duplicate completed checkpoint result: {condition_id}:{trajectory_id}"
            )
        completed[key] = value
    inventory: list[CheckpointPair] = []
    for key in sorted(completed):
        observable_path, label_path = observable_paths.get(key), label_paths.get(key)
        if observable_path is None or label_path is None:
            raise ArtifactConflict(f"incomplete checkpoint artifact pair: {key[0]}:{key[1]}")
        inventory.append(
            _verified_checkpoint_pair(
                root,
                observable_path,
                label_path,
                completed[key],
                resolved_config_hash,
                "pilot",
            )
        )
    sealed_commitment = manifest.get("sealed_commitment")
    has_sealed_pairs = any(pair.relative_path.parts[2] == "sealed_test" for pair in inventory)
    if sealed_commitment is not None and not _is_sha256_commitment(sealed_commitment):
        raise ArtifactConflict("checkpoint sealed commitment is invalid")
    if has_sealed_pairs and not _is_sha256_commitment(sealed_commitment):
        raise ArtifactConflict("checkpoint sealed commitment is required")
    commitment_path = root / "data" / "manifests" / "sealed_commitment.json"
    if commitment_path.exists():
        try:
            source_commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactConflict("checkpoint sealed commitment is invalid") from error
        if source_commitment != sealed_commitment:
            raise ArtifactConflict("checkpoint sealed commitment mismatch")
    return tuple(inventory)


_CHECKPOINT_ARTIFACT_FILES = ("arrays.npz", "metadata.json", "SHA256SUMS")


def _checkpoint_file_set(inventory: Sequence[CheckpointPair]) -> set[Path]:
    files = {Path("run_manifest.json")}
    for pair in inventory:
        for directory in (pair.relative_path, pair.label_relative_path):
            files.update(directory / name for name in _CHECKPOINT_ARTIFACT_FILES)
    return files


def _copy_checkpoint_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ArtifactConflict(f"invalid checkpoint file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _fsync_file(target)


def _checkpoint_export_matches(
    source: Path,
    destination: Path,
    inventory: tuple[CheckpointPair, ...],
) -> bool:
    try:
        if destination.is_symlink() or not destination.is_dir():
            return False
        entries = tuple(destination.rglob("*"))
        if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
            return False
        if (destination / "run_manifest.json").read_bytes() != (
            source / "run_manifest.json"
        ).read_bytes():
            return False
        if inventory_checkpoint(destination) != inventory:
            return False
        actual_files = {path.relative_to(destination) for path in entries if path.is_file()}
        return actual_files == _checkpoint_file_set(inventory)
    except (OSError, TypeError, ValueError):
        return False


def _publish_checkpoint_export(
    staging: Path,
    destination: Path,
    source: Path,
    inventory: tuple[CheckpointPair, ...],
) -> None:
    try:
        _rename_no_replace(staging, destination)
    except FileExistsError as error:
        if _checkpoint_export_matches(source, destination, inventory):
            shutil.rmtree(staging)
            return
        raise ArtifactConflict(f"checkpoint export conflict: {destination}") from error
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
        lock_path = destination.with_name(f".{destination.name}.publish.lock")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as lock_error:
            raise ArtifactConflict(
                f"checkpoint export publication is busy: {destination}"
            ) from lock_error
        try:
            os.close(descriptor)
            if destination.exists():
                if _checkpoint_export_matches(source, destination, inventory):
                    shutil.rmtree(staging)
                    return
                raise ArtifactConflict(f"checkpoint export conflict: {destination}")
            os.replace(staging, destination)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    _fsync_directory(destination.parent)


def export_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_config_hash: str | None = None,
    expected_provenance: ManifestProvenance | None = None,
) -> Path:
    """Atomically export a manifest and its verified artifact pairs."""
    inventory = inventory_checkpoint(
        source,
        expected_config_hash=expected_config_hash,
        expected_provenance=expected_provenance,
    )
    if destination.exists():
        if _checkpoint_export_matches(source, destination, inventory):
            return destination
        raise ArtifactConflict(f"checkpoint export conflict: {destination}")
    staging = _staging_directory(destination)
    try:
        _copy_checkpoint_file(source / "run_manifest.json", staging / "run_manifest.json")
        for pair in inventory:
            for relative in (pair.relative_path, pair.label_relative_path):
                for name in _CHECKPOINT_ARTIFACT_FILES:
                    _copy_checkpoint_file(source / relative / name, staging / relative / name)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        if inventory_checkpoint(staging) != inventory:
            raise ArtifactConflict("checkpoint export verification mismatch")
        _publish_checkpoint_export(staging, destination, source, inventory)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_sealed_commitment(private_path: Path, commitment_path: Path) -> str:
    """Commit canonical private seed bytes without copying the seed into public data."""
    private_bytes = private_path.read_bytes()
    digest = hashlib.sha256(private_bytes).hexdigest()
    commitment_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(commitment_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"sealed commitment exists: {commitment_path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json({"algorithm": "sha256", "digest": digest}))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            commitment_path.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(commitment_path.parent)
    return digest


def load_sealed_seed(
    private_path: Path,
    commitment_path: Path,
    *,
    purpose: Literal["development", "sealed_evaluation"],
) -> int:
    if purpose != "sealed_evaluation":
        raise PermissionError("sealed evaluation only")
    private_bytes = private_path.read_bytes()
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if not isinstance(commitment, Mapping) or hashlib.sha256(
        private_bytes
    ).hexdigest() != commitment.get("digest"):
        raise ValueError("sealed commitment mismatch")
    private = json.loads(private_bytes)
    if not isinstance(private, Mapping) or isinstance(private.get("root_seed"), bool):
        raise TypeError("invalid sealed manifest")
    return int(private["root_seed"])


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(path: Path, manifest: Mapping[str, object]) -> str:
    _atomic_write(path, lambda temporary: temporary.write_bytes(_canonical_json(dict(manifest))))
    return _sha256(path)


def write_weight_table(
    path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> str:
    staging = _staging_directory(path)
    artifact_hash = ""
    owned_identity: tuple[int, int] | None = None
    try:
        _write_deterministic_npz(staging / "arrays.npz", arrays)
        _write_json(staging / "metadata.json", dict(metadata))
        artifact_hash, owned_identity = _publish_directory(staging, path)
        _fsync_directory(path.parent)
        return artifact_hash
    except BaseException:
        if owned_identity is not None:
            _remove_owned_artifact(path, artifact_hash, owned_identity)
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    def write(temporary: Path) -> None:
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary)

    _atomic_write(path, write)
    return _sha256(path)
