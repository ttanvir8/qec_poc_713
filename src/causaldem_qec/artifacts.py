from __future__ import annotations

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


def _publish_directory(staging: Path, target: Path) -> str:
    artifact_hash = _write_sums(staging)
    if target.exists():
        if verify_artifact(target) == artifact_hash:
            shutil.rmtree(staging)
            return artifact_hash
        raise FileExistsError(f"artifact conflict: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    _fsync_directory(target.parent)
    return artifact_hash


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
    observable_path = (
        root / "data" / "observable" / job.split / job.condition_id / str(job.trajectory_id)
    )
    label_path = root / "data" / "labels" / job.split / job.condition_id / str(job.trajectory_id)
    pair_id = _pair_id(job, metadata)
    observable_staging = _staging_directory(observable_path)
    label_staging = _staging_directory(label_path)
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
        _write_sums(observable_staging)
        _write_sums(label_staging)
        verify_artifact(observable_staging)
        verify_artifact(label_staging)
        _publish_directory(observable_staging, observable_path)
        _publish_directory(label_staging, label_path)
    except BaseException:
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
    try:
        _write_deterministic_npz(staging / "arrays.npz", arrays)
        _write_json(staging / "metadata.json", dict(metadata))
        return _publish_directory(staging, path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    def write(temporary: Path) -> None:
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary)

    _atomic_write(path, write)
    return _sha256(path)
