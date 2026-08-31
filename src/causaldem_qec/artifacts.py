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
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from causaldem_qec.core import (
    LabelTrajectory,
    ObservableTrajectory,
    TrajectoryJob,
    validate_labels,
    validate_observable,
)

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


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
        raise FileExistsError(f"artifact conflict: {target}")
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
        raise ValueError("incomplete artifact pair")
    if existing_observable is None:
        return False
    if existing_observable != observable_hash or existing_labels != label_hash:
        raise FileExistsError(f"artifact conflict: {observable_path}")
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
