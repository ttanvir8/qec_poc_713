"""Chunked, pilot-only dataset reporting.

The reporting lane is deliberately offline: it reads labels only while making
pre-model diagnostics and never exposes them through observable loaders.
"""

from __future__ import annotations

import json
import struct
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib
import numpy as np
from scipy import signal  # type: ignore[import-untyped]
from scipy.linalg import solve_toeplitz  # type: ignore[import-untyped]

from causaldem_qec.artifacts import verify_artifact

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DATASET_EDA_SECTIONS = (
    "inventory_and_integrity",
    "detector_and_logical_rates",
    "physical_and_class_probabilities",
    "trajectory_views",
    "temporal_spectra",
    "spatial_correlations",
    "observation_corruption",
    "parity_theory_check",
    "noncommutation_gap",
    "canonical_catalog",
    "split_isolation",
)
_TRUTH_SECTIONS = frozenset(
    {
        "physical_and_class_probabilities",
        "temporal_spectra",
        "spatial_correlations",
        "parity_theory_check",
        "noncommutation_gap",
        "canonical_catalog",
    }
)
_PILOT_NOTICE = (
    "PILOT / NOT FINAL — final support/non-support and condition-level statistical "
    "claims require the deferred full production dataset."
)


@dataclass(frozen=True, slots=True)
class EdaSection:
    uses_simulator_truth: bool
    source_artifact_hashes: tuple[str, ...]
    row_count: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class EdaIndex:
    sections: tuple[str, ...]
    truth_sections: frozenset[str]
    output_paths: Mapping[str, Path]
    max_loaded_rounds: int
    section_records: Mapping[str, EdaSection]


@dataclass(frozen=True, slots=True)
class _Pair:
    condition_id: str
    trajectory_id: int
    split: str
    observable: Path
    labels: Path
    hashes: tuple[str, str]
    metadata: Mapping[str, object]


def validate_inventory(keys: Sequence[tuple[str, int]]) -> None:
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trajectory id")


def _manifest(root: Path) -> Mapping[str, object]:
    try:
        value = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pilot EDA requires a pilot run manifest") from error
    if not isinstance(value, dict) or value.get("dataset_profile") != "pilot":
        raise ValueError("pilot EDA requires a pilot-profile dataset root")
    return MappingProxyType(value)


def _pairs(root: Path, manifest: Mapping[str, object]) -> tuple[_Pair, ...]:
    pairs: list[_Pair] = []
    for observable in sorted((root / "data" / "observable").glob("*/*/*")):
        if not observable.is_dir():
            continue
        relative = observable.relative_to(root / "data" / "observable")
        split, condition_id, trajectory = relative.parts
        try:
            trajectory_id = int(trajectory)
        except ValueError as error:
            raise ValueError(f"invalid trajectory id: {observable}") from error
        labels = root / "data" / "labels" / split / condition_id / trajectory
        if not labels.is_dir():
            raise ValueError("incomplete artifact pair")
        observable_hash, label_hash = verify_artifact(observable), verify_artifact(labels)
        try:
            label_document = json.loads((labels / "metadata.json").read_text(encoding="utf-8"))
            metadata = label_document["metadata"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid label metadata") from error
        if not isinstance(metadata, dict):
            raise TypeError("invalid label metadata")
        pairs.append(
            _Pair(
                condition_id,
                trajectory_id,
                split,
                observable,
                labels,
                (observable_hash, label_hash),
                MappingProxyType(metadata),
            )
        )
    validate_inventory([(pair.condition_id, pair.trajectory_id) for pair in pairs])
    results = manifest.get("results")
    result_keys = (
        [
            (result.get("condition_id"), result.get("trajectory_id"))
            for result in results
            if isinstance(result, Mapping)
        ]
        if isinstance(results, list)
        else []
    )
    if (
        not isinstance(results, list)
        or any(
            not isinstance(result, Mapping) or result.get("completed") is not True
            for result in results
        )
        or len(results) != len(pairs)
        or set(result_keys) != {(pair.condition_id, pair.trajectory_id) for pair in pairs}
        or len(result_keys) != len(set(result_keys))
    ):
        raise ValueError("pilot EDA requires a complete pilot root")
    if not pairs:
        raise ValueError("pilot EDA requires at least one verified trajectory")
    return tuple(pairs)


def _member_layout(path: Path, member: str) -> tuple[np.dtype[Any], tuple[int, ...], int]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("chunked EDA requires stored NumPy artifact arrays")
        with path.open("rb") as handle:
            handle.seek(info.header_offset + 26)
            name_length, extra_length = struct.unpack("<HH", handle.read(4))
            payload = info.header_offset + 30 + name_length + extra_length
            handle.seek(payload)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            array_offset = handle.tell()
    if fortran or not shape:
        raise ValueError("invalid chunkable artifact array")
    return dtype, tuple(shape), array_offset


def _chunked(path: Path, member: str, start: int, stop: int) -> np.ndarray:
    dtype, shape, header = _member_layout(path, member)
    if stop > shape[0]:
        raise ValueError("invalid chunk bounds")
    # ZIP_STORED .npy members are directly memmappable; only this row slice is read.
    array = np.memmap(path, dtype=dtype, mode="r", offset=header, shape=shape)
    return np.asarray(array[start:stop]).copy()


def _rounds(path: Path) -> int:
    _, shape, _ = _member_layout(path, "global_round")
    return shape[0]


def _iter_verified_chunks(
    pairs: Sequence[_Pair], chunk_rounds: int
) -> Iterator[tuple[_Pair, Mapping[str, np.ndarray]]]:
    if chunk_rounds <= 0:
        raise ValueError("chunk_rounds must be positive")
    fields = (
        "detector_bits_packed",
        "detector_valid_packed",
        "global_round",
        "block",
        "circuit_phase",
    )
    label_fields = ("component_probability", "latent_factor", "class_probability")
    for pair in pairs:
        total = _rounds(pair.observable / "arrays.npz")
        detector_count = int(_chunked(pair.observable / "arrays.npz", "detector_shape", 0, 2)[1])
        for start in range(0, total, chunk_rounds):
            stop = min(total, start + chunk_rounds)
            values = {
                name: _chunked(pair.observable / "arrays.npz", name, start, stop) for name in fields
            }
            values.update(
                {
                    name: _chunked(pair.labels / "arrays.npz", name, start, stop)
                    for name in label_fields
                }
            )
            values["detector_count"] = np.asarray([detector_count], dtype=np.int64)
            yield pair, MappingProxyType(values)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _pacf(values: np.ndarray, order: int = 8) -> list[float]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    if centered.size < 3 or not np.any(centered):
        return [0.0]
    size = min(order, centered.size - 1)
    acf = np.correlate(centered, centered, mode="full")[centered.size - 1 : centered.size + size]
    acf /= acf[0]
    result = [1.0]
    for lag in range(1, size + 1):
        result.append(float(solve_toeplitz((acf[:lag], acf[:lag]), acf[1 : lag + 1])[-1]))
    return result


def _figure(output_root: Path, profile_hash: str, series: np.ndarray) -> Path:
    path = output_root / "trajectory_views.png"
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot(series[: min(series.size, 512)])
    axis.set_title(
        "PILOT / NOT FINAL — observable parity sample — "
        f"{profile_hash[:12]}\nFinal claims require the deferred full production dataset."
    )
    axis.set_xlabel("round")
    axis.set_ylabel("parity")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _condition_parts(condition_id: str) -> tuple[str, int, str]:
    circuit, dynamics = condition_id.split("__", maxsplit=1)
    try:
        distance = int(circuit.rsplit("d", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid condition id: {condition_id}") from error
    return circuit, distance, dynamics


def _mean_record(total: float, count: int) -> dict[str, float | int]:
    return {"events": int(total), "opportunities": int(count), "rate": total / max(count, 1)}


def _update_moments(moments: list[float], values: np.ndarray) -> None:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flattened.size:
        return
    moments[0] += float(flattened.size)
    moments[1] += float(flattened.sum())
    moments[2] += float(np.dot(flattened, flattened))
    moments[3] = min(moments[3], float(flattened.min()))
    moments[4] = max(moments[4], float(flattened.max()))


def _moments_summary(moments: Sequence[float]) -> dict[str, float | int]:
    count, total, squared, minimum, maximum = moments
    mean = total / max(count, 1.0)
    return {
        "count": int(count),
        "mean": mean,
        "std": float(np.sqrt(max(squared / max(count, 1.0) - mean * mean, 0.0))),
        "minimum": minimum if count else 0.0,
        "maximum": maximum if count else 0.0,
    }


def _run_lengths(mask: np.ndarray) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    padded = np.concatenate((np.array([False]), values, np.array([False])))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [int(stop - start) for start, stop in zip(edges[::2], edges[1::2], strict=True)]


def _continued_run_lengths(mask: np.ndarray, carry: int) -> tuple[list[int], int]:
    completed: list[int] = []
    run = carry
    for value in np.asarray(mask, dtype=bool):
        if value:
            run += 1
        elif run:
            completed.append(run)
            run = 0
    return completed, run


def _correlation(
    sum_x: float, sum_y: float, sum_xy: float, sum_x2: float, sum_y2: float, n: int
) -> float:
    if n < 2:
        return 0.0
    numerator = n * sum_xy - sum_x * sum_y
    denominator = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
    return float(numerator / np.sqrt(denominator)) if denominator > 0.0 else 0.0


def _series_summary(values: np.ndarray) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    if series.size < 4:
        return {"mean_power": 0.0, "mean_acf": 0.0, "mean_pacf": 0.0}
    _, power = signal.periodogram(series)
    centered = series - float(np.mean(series))
    denominator = float(np.dot(centered, centered))
    acf = (
        [
            float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            for lag in range(1, min(9, len(centered)))
        ]
        if denominator
        else []
    )
    pacf = _pacf(series)[1:]
    return {
        "mean_power": float(np.mean(power)),
        "mean_acf": float(np.mean(acf)) if acf else 0.0,
        "mean_pacf": float(np.mean(pacf)) if pacf else 0.0,
    }


def _metadata_mapping(metadata: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = metadata.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _metadata_sequence(metadata: Mapping[str, object], key: str) -> Sequence[object]:
    value = metadata.get(key, ())
    return value if isinstance(value, (list, tuple)) else ()


def render_data_card(
    output_root: Path, *, profile_hash: str, inventory: Mapping[str, object]
) -> Path:
    path = output_root / "data_card.md"
    geometry = inventory.get("geometry", {})
    circuits = inventory.get("circuits", [])
    physical_bounds = inventory.get("physical_error_bounds", [])
    schemas = inventory.get("schemas", [])
    splits = inventory.get("splits", {})
    path.write_text(
        "# CausalDEM-QEC pilot data card\n\n"
        f"{_PILOT_NOTICE}\n\n"
        f"Dataset profile/config hash: `{profile_hash}`. Package hash: "
        f"`{sha256(Path(__file__).read_bytes()).hexdigest()}`.\n\n"
        "Generation law: round-varying simulator trajectories with exact-XOR canonical classes. "
        "Physical errors, detector observations, and offline labels remain separated.\n\n"
        f"Circuits: `{json.dumps(circuits, sort_keys=True)}`.\n\n"
        f"Geometry: `{json.dumps(geometry, sort_keys=True)}`.\n\n"
        f"Splits: `{json.dumps(splits, sort_keys=True)}`.\n\n"
        f"Schemas: `{json.dumps(schemas, sort_keys=True)}`.\n\n"
        f"Physical-error bounds: `{json.dumps(physical_bounds, sort_keys=True)}`.\n\n"
        "Circuits, trajectory/episode/block sizes, physical-error bounds, and split counts are "
        "recorded in the manifest. Intended use is pipeline validation only; forbidden leakage includes labels, sealed seeds, "
        "and cross-partition normalizers. Independent replicates are trajectory IDs.\n\n"
        "Package/config hashes are recorded with artifacts. Known limitations: this is a "
        "scaled pilot, not a basis for final scientific conclusions. Reserve 80–100 GiB for the pilot.\n\n"
        f"Inventory: `{json.dumps(dict(inventory), sort_keys=True)}`\n",
        encoding="utf-8",
    )
    return path


def render_validation_report(
    output_root: Path,
    *,
    profile_hash: str,
    inventory: Mapping[str, object],
    gate_results: Sequence[object] = (),
) -> Path:
    path = output_root / "validation_report.json"
    gates: list[dict[str, object]] = []
    by_id = {str(getattr(item, "gate_id", "")): item for item in gate_results}
    for number in range(1, 13):
        gate_id = f"DQ{number:02d}"
        result = by_id.get(gate_id)
        evidence = (
            dict(getattr(result, "evidence", inventory)) if result is not None else dict(inventory)
        )
        gates.append(
            {
                "gate_id": gate_id,
                "status": (
                    "not_run"
                    if gate_id == "DQ08" or result is None
                    else str(getattr(getattr(result, "status", None), "value", "not_run"))
                ),
                "evidence": evidence,
                "thresholds": {key: value for key, value in evidence.items() if "tolerance" in key},
                "affected_conditions": list(getattr(result, "affected_conditions", ())),
                "failure_records": []
                if result is None
                else (
                    []
                    if str(getattr(getattr(result, "status", None), "value", "")) == "pass"
                    else [evidence]
                ),
            }
        )
    _write_json(
        path,
        {
            "scientific_status": "PILOT / NOT FINAL",
            "dataset_profile_hash": profile_hash,
            "scientific_limitation": _PILOT_NOTICE,
            "gates": gates,
        },
    )
    return path


def build_dataset_eda(
    input_root: Path,
    output_root: Path,
    sample_seed: int,
    chunk_rounds: int = 4096,
    gate_results: Sequence[object] = (),
) -> EdaIndex:
    manifest = _manifest(input_root)
    pairs = _pairs(input_root, manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    profile_hash = str(manifest.get("resolved_config_hash", "unknown"))
    hashes = tuple(sorted(hash_value for pair in pairs for hash_value in pair.hashes))
    generation = manifest.get("generation")
    generation_record = generation if isinstance(generation, Mapping) else {}
    splits = {
        split: sum(pair.split == split for pair in pairs)
        for split in sorted({pair.split for pair in pairs})
    }
    geometry = {
        "burn_in_rounds": generation_record.get("burn_in_rounds"),
        "scored_rounds": generation_record.get("scored_rounds"),
        "episode_rounds": sorted(
            {
                value
                for pair in pairs
                if isinstance(value := pair.metadata.get("episode_rounds"), int)
            }
        ),
        "block_rounds": sorted(
            {value for pair in pairs if isinstance(value := pair.metadata.get("block_rounds"), int)}
        ),
    }
    bounds = sorted(
        {
            tuple(item)
            for pair in pairs
            for item in _metadata_sequence(
                _metadata_mapping(pair.metadata, "generation_law"), "component_bounds"
            )
            if isinstance(item, (list, tuple)) and len(item) == 3
        }
    )
    inventory: dict[str, object] = {
        "trajectories": len(pairs),
        "circuits": sorted({pair.condition_id.split("__", maxsplit=1)[0] for pair in pairs}),
        "conditions": sorted({pair.condition_id for pair in pairs}),
        "splits": splits,
        "generation": generation_record,
        "schemas": ["observable-v1", "labels-v1"],
        "geometry": geometry,
        "physical_error_bounds": bounds,
        "disk_bytes": sum(
            file.stat().st_size
            for pair in pairs
            for file in (pair.observable / "arrays.npz", pair.labels / "arrays.npz")
        ),
        "failed_jobs": 0,
        "duplicate_ids": 0,
        "profile_hash": profile_hash,
    }
    totals: dict[str, float] = defaultdict(float)
    rate_buckets: dict[str, dict[str, list[int]]] = {
        name: defaultdict(lambda: [0, 0])
        for name in ("circuit", "distance", "phase", "detector_role", "dynamics", "time")
    }
    logical_buckets: dict[str, dict[str, list[int]]] = {
        name: defaultdict(lambda: [0, 0]) for name in ("circuit", "distance", "dynamics", "time")
    }
    spectra: dict[str, list[dict[str, float]]] = defaultdict(list)
    component_moments = [0.0, 0.0, 0.0, float("inf"), float("-inf")]
    class_moments = [0.0, 0.0, 0.0, float("inf"), float("-inf")]
    heterogeneity = [0.0, 0.0, 0.0, float("inf"), float("-inf")]
    logical_events = 0
    logical_total = 0
    covariance_signs: dict[str, int] = defaultdict(int)
    block_class_sums: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    selected = min(
        pairs,
        key=lambda pair: sha256(
            f"{sample_seed}:{pair.condition_id}:{pair.trajectory_id}".encode()
        ).digest(),
    )
    selected_parity: list[float] = []
    selected_episode: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    selected_block: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    selected_missing: list[int] = []
    selected_regime: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    missing_runs: list[int] = []
    open_missing_runs: dict[tuple[str, int], int] = defaultdict(int)
    selected_open_missing = 0
    physical_independence: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    regime_blocks: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    for pair, values in _iter_verified_chunks(pairs, chunk_rounds):
        detector_count = int(values["detector_count"][0])
        packed = values["detector_bits_packed"]
        bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :detector_count].astype(bool)
        valid = np.unpackbits(values["detector_valid_packed"], axis=1, bitorder="little")[
            :, :detector_count
        ].astype(bool)
        queried = np.all(valid, axis=1)
        parity = np.logical_xor.reduce(bits[queried], axis=1).astype(float)
        class_probability = values["class_probability"][queried]
        # Exact XOR composition, not an OR probability.
        theory = 0.5 * (1.0 - np.prod(1.0 - 2.0 * class_probability, axis=1))
        for block in np.unique(values["block"][queried]):
            class_values = class_probability[values["block"][queried] == block]
            key = (pair.condition_id, int(block))
            previous, count = block_class_sums.get(key, (np.zeros(class_values.shape[1]), 0))
            block_class_sums[key] = (previous + class_values.sum(axis=0), count + len(class_values))
        circuit, distance, dynamics = _condition_parts(pair.condition_id)
        time_fraction = values["global_round"] / max(_rounds(pair.observable / "arrays.npz"), 1)
        time_strata = np.where(
            time_fraction < 1 / 3, "early", np.where(time_fraction < 2 / 3, "middle", "late")
        )
        for name, bucket_key, event, count in (
            ("circuit", circuit, int(bits.sum()), bits.size),
            ("distance", f"d{distance}", int(bits.sum()), bits.size),
            ("dynamics", dynamics, int(bits.sum()), bits.size),
        ):
            rate_buckets[name][bucket_key][0] += event
            rate_buckets[name][bucket_key][1] += count
        roles = _chunked(pair.observable / "arrays.npz", "detector_role", 0, detector_count)
        for role in np.unique(roles):
            mask = roles == role
            rate_buckets["detector_role"][str(int(role))][0] += int(bits[:, mask].sum())
            rate_buckets["detector_role"][str(int(role))][1] += int(mask.sum() * len(bits))
        for phase in np.unique(values["circuit_phase"]):
            mask = values["circuit_phase"] == phase
            rate_buckets["phase"][str(int(phase))][0] += int(bits[mask].sum())
            rate_buckets["phase"][str(int(phase))][1] += int(mask.sum() * detector_count)
        for stratum in ("early", "middle", "late"):
            mask = time_strata == stratum
            rate_buckets["time"][stratum][0] += int(bits[mask].sum())
            rate_buckets["time"][stratum][1] += int(mask.sum() * detector_count)
        totals["rounds"] += len(values["global_round"])
        totals["valid"] += int(valid.sum())
        totals["detectors"] += valid.size
        totals["bits"] += int(bits.sum())
        totals["parity"] += float(parity.sum())
        totals["theory"] += float(theory.sum())
        totals["queried"] += len(parity)
        _update_moments(component_moments, values["component_probability"])
        _update_moments(class_moments, values["class_probability"])
        _update_moments(heterogeneity, np.ptp(values["component_probability"], axis=1))
        spectra["latent_factor"].append(_series_summary(values["latent_factor"].mean(axis=1)))
        spectra["class_probability"].append(
            _series_summary(values["class_probability"].mean(axis=1))
        )
        spectra["observable_parity"].append(_series_summary(parity))
        covariance = np.cov(bits.astype(float), rowvar=False) if len(bits) > 1 else np.zeros((2, 2))
        covariance_signs["positive"] += int((covariance > 0).sum())
        covariance_signs["negative"] += int((covariance < 0).sum())
        component_covariance = (
            np.cov(values["component_probability"], rowvar=False)
            if values["component_probability"].shape[1] > 1
            else np.zeros((1, 1))
        )
        covariance_signs["component_positive"] += int((component_covariance > 0).sum())
        covariance_signs["component_negative"] += int((component_covariance < 0).sum())
        missing = ~np.all(valid, axis=1)
        pair_key = (pair.condition_id, pair.trajectory_id)
        completed_runs, open_missing_runs[pair_key] = _continued_run_lengths(
            missing, open_missing_runs[pair_key]
        )
        missing_runs.extend(completed_runs)
        mean_physical = values["component_probability"].mean(axis=1)
        physical_independence[0] += float(missing.sum())
        physical_independence[1] += float(mean_physical.sum())
        physical_independence[2] += float(np.dot(missing, mean_physical))
        physical_independence[3] += float(np.dot(missing, missing))
        physical_independence[4] += float(np.dot(mean_physical, mean_physical))
        physical_independence[5] += len(missing)
        for block in np.unique(values["block"]):
            mask = values["block"] == block
            bucket = regime_blocks[(pair.condition_id, pair.trajectory_id)][int(block)]
            bucket[0] += float(mean_physical[mask].sum())
            bucket[1] += int(mask.sum())
        if pair == selected:
            selected_parity.extend(parity[: max(0, 512 - len(selected_parity))])
            selected_completed, selected_open_missing = _continued_run_lengths(
                missing, selected_open_missing
            )
            selected_missing.extend(selected_completed)
            for episode in np.unique(values["global_round"] // 32):
                mask = values["global_round"] // 32 == episode
                bucket = selected_episode[int(episode)]
                bucket[0] += float(np.logical_xor.reduce(bits[mask], axis=1).sum())
                bucket[1] += int(mask.sum())
            for block in np.unique(values["block"]):
                mask = values["block"] == block
                bucket = selected_block[int(block)]
                bucket[0] += float(np.logical_xor.reduce(bits[mask], axis=1).sum())
                bucket[1] += int(mask.sum())
                regime = selected_regime[int(block)]
                regime[0] += float(mean_physical[mask].sum())
                regime[1] += int(mask.sum())
    for pair in pairs:
        logical_count = _member_layout(pair.observable / "arrays.npz", "logical_observable")[1][0]
        circuit, distance, dynamics = _condition_parts(pair.condition_id)
        for start in range(0, logical_count, max(chunk_rounds // 32, 1)):
            logical = _chunked(
                pair.observable / "arrays.npz",
                "logical_observable",
                start,
                min(logical_count, start + max(chunk_rounds // 32, 1)),
            )
            logical_events += int(logical.sum())
            logical_total += logical.size
            for name, bucket_key in (
                ("circuit", circuit),
                ("distance", f"d{distance}"),
                ("dynamics", dynamics),
            ):
                logical_buckets[name][bucket_key][0] += int(logical.sum())
                logical_buckets[name][bucket_key][1] += logical.size
            episode_fraction = np.arange(start, start + logical.size) / max(logical_count, 1)
            episode_strata = np.where(
                episode_fraction < 1 / 3,
                "early",
                np.where(episode_fraction < 2 / 3, "middle", "late"),
            )
            for stratum in ("early", "middle", "late"):
                mask = episode_strata == stratum
                logical_buckets["time"][stratum][0] += int(logical[mask].sum())
                logical_buckets["time"][stratum][1] += int(mask.sum())
    missing_runs.extend(length for length in open_missing_runs.values() if length)
    if selected_open_missing:
        selected_missing.append(selected_open_missing)
    totals["block_theory"] = sum(
        float(0.5 * (1.0 - np.prod(1.0 - 2.0 * (values / count))) * count)
        for values, count in block_class_sums.values()
    )
    rates = {
        "detector_rate": totals["bits"] / max(int(totals["detectors"]), 1),
        "valid_rate": totals["valid"] / max(int(totals["detectors"]), 1),
        "mean_component_probability": float(_moments_summary(component_moments)["mean"]),
        "mean_class_probability": float(_moments_summary(class_moments)["mean"]),
        "empirical_parity": totals["parity"] / max(int(totals["queried"]), 1),
        "theoretical_parity": totals["theory"] / max(int(totals["queried"]), 1),
        "block_average_theoretical_parity": totals["block_theory"] / max(int(totals["queried"]), 1),
        "positive_covariances": covariance_signs["positive"],
        "negative_covariances": covariance_signs["negative"],
    }
    _figure(output_root, profile_hash, np.asarray(selected_parity, dtype=float))
    spectral_details = {
        name: {
            metric: float(np.mean([item[metric] for item in items])) if items else 0.0
            for metric in ("mean_power", "mean_acf", "mean_pacf")
        }
        for name, items in spectra.items()
    }
    dwell_lengths: list[int] = []
    for blocks in regime_blocks.values():
        means = np.asarray([item[0] / max(item[1], 1.0) for _, item in sorted(blocks.items())])
        dwell_lengths.extend(_run_lengths(means > np.median(means)))
    catalog_by_hash = {
        str(pair.metadata.get("canonical_catalog_hash", "fixture-catalog")): _metadata_mapping(
            pair.metadata, "canonical_catalog"
        )
        for pair in pairs
    }
    catalog_summaries = [summary for summary in catalog_by_hash.values() if summary]
    class_counts = sorted(
        {
            value
            for summary in catalog_summaries
            if isinstance(value := summary.get("class_count"), int)
        }
    )
    duplicate_sizes = sorted(
        int(size)
        for summary in catalog_summaries
        for size in _metadata_sequence(summary, "duplicate_sizes")
        if isinstance(size, int)
    )
    catalog_mass = {
        name: float(
            np.mean(
                [
                    float(value)
                    for summary in catalog_summaries
                    if isinstance(value := summary.get(name), (int, float))
                ]
            )
        )
        if any(isinstance(summary.get(name), (int, float)) for summary in catalog_summaries)
        else 0.0
        for name in ("graphlike_mass", "adaptable_mass", "ambiguous_logical_mass", "hyperedge_mass")
    }
    configured_contamination = sorted(
        {
            float(value)
            for pair in pairs
            if isinstance(
                value := _metadata_mapping(pair.metadata, "generation_law").get(
                    "observation_flip_probability", 0.0
                ),
                (int, float),
            )
        }
    )
    details: dict[str, Mapping[str, object]] = {
        "inventory_and_integrity": inventory,
        "detector_and_logical_rates": {
            "by_circuit": {
                key: _mean_record(*value) for key, value in rate_buckets["circuit"].items()
            },
            "by_distance": {
                key: _mean_record(*value) for key, value in rate_buckets["distance"].items()
            },
            "by_phase": {key: _mean_record(*value) for key, value in rate_buckets["phase"].items()},
            "by_detector_role": {
                key: _mean_record(*value) for key, value in rate_buckets["detector_role"].items()
            },
            "by_dynamics": {
                key: _mean_record(*value) for key, value in rate_buckets["dynamics"].items()
            },
            "by_time_stratum": {
                key: _mean_record(*value) for key, value in rate_buckets["time"].items()
            },
            "logical_rate": _mean_record(logical_events, logical_total),
            "logical_by_circuit": {
                key: _mean_record(*value) for key, value in logical_buckets["circuit"].items()
            },
            "logical_by_distance": {
                key: _mean_record(*value) for key, value in logical_buckets["distance"].items()
            },
            "logical_by_dynamics": {
                key: _mean_record(*value) for key, value in logical_buckets["dynamics"].items()
            },
            "logical_by_time_stratum": {
                key: _mean_record(*value) for key, value in logical_buckets["time"].items()
            },
            "logical_phase_and_detector_role": "not applicable: logical observables are episode-level",
        },
        "physical_and_class_probabilities": {
            "component_distribution": _moments_summary(component_moments),
            "class_distribution": _moments_summary(class_moments),
            "heterogeneity": _moments_summary(heterogeneity),
            "boundary_checks": {
                "component_values_in_unit_interval": component_moments[3] >= 0.0
                and component_moments[4] < 0.5,
                "class_values_in_unit_interval": class_moments[3] >= 0.0 and class_moments[4] < 0.5,
                "declared_physical_bounds": bounds,
            },
        },
        "trajectory_views": {
            "selection": {
                "condition_id": selected.condition_id,
                "trajectory_id": selected.trajectory_id,
                "method": "sha256",
            },
            "figure": "trajectory_views.png",
            "long": selected_parity,
            "episode": {
                str(key): value[0] / max(value[1], 1.0) for key, value in selected_episode.items()
            },
            "block": {
                str(key): value[0] / max(value[1], 1.0) for key, value in selected_block.items()
            },
            "burst": selected_missing,
            "regime": {
                str(key): value[0] / max(value[1], 1.0) for key, value in selected_regime.items()
            },
        },
        "temporal_spectra": spectral_details,
        "spatial_correlations": {
            "detector_positive_covariances": covariance_signs["positive"],
            "detector_negative_covariances": covariance_signs["negative"],
            "component_positive_covariances": covariance_signs["component_positive"],
            "component_negative_covariances": covariance_signs["component_negative"],
        },
        "observation_corruption": {
            "missing_run_lengths": missing_runs,
            "contamination": {"configured_flip_probabilities": configured_contamination},
            "burst_durations": missing_runs,
            "regime_dwell_times": dwell_lengths,
            "physical_independence": {
                "missingness_component_mean_correlation": _correlation(
                    physical_independence[0],
                    physical_independence[1],
                    physical_independence[2],
                    physical_independence[3],
                    physical_independence[4],
                    int(physical_independence[5]),
                ),
                "observed_missing_rate": 1.0 - rates["valid_rate"],
            },
        },
        "parity_theory_check": {
            "empirical": rates["empirical_parity"],
            "theoretical": rates["theoretical_parity"],
        },
        "noncommutation_gap": {
            "round_mean_minus_block_average": rates["theoretical_parity"]
            - rates["block_average_theoretical_parity"]
        },
        "canonical_catalog": {
            "class_counts": class_counts,
            "duplicate_sizes": duplicate_sizes,
            "graphlike_mass": catalog_mass["graphlike_mass"],
            "adaptable_mass": catalog_mass["adaptable_mass"],
            "ambiguous_mass": catalog_mass["ambiguous_logical_mass"],
            "hyperedge_mass": catalog_mass["hyperedge_mass"],
            "available_singular_spectra": [],
        },
        "split_isolation": {
            "split_counts": splits,
            "trajectory_ids": {
                split: [
                    [pair.condition_id, pair.trajectory_id] for pair in pairs if pair.split == split
                ]
                for split in splits
            },
            "trajectory_ids_disjoint": len(
                {(pair.condition_id, pair.trajectory_id) for pair in pairs}
            )
            == len(pairs),
            "seed_commitments": sorted(
                {str(pair.metadata.get("public_seed_commitment", "")) for pair in pairs}
            ),
            "artifact_hashes": hashes,
            "normalizer_isolation": {
                "fitted": False,
                "evidence": "dataset-stage EDA fits no normalizer",
            },
            "recurrent_state_isolation": {
                "fitted": False,
                "evidence": "dataset-stage EDA has no recurrent state",
            },
        },
    }
    paths: dict[str, Path] = {}
    records: dict[str, EdaSection] = {}
    for section in DATASET_EDA_SECTIONS:
        path = output_root / f"{section}.json"
        payload: dict[str, object] = {
            "section": section,
            "scientific_status": "PILOT / NOT FINAL",
            "scientific_limitation": _PILOT_NOTICE,
            "dataset_profile_hash": profile_hash,
            "uses_simulator_truth": section in _TRUTH_SECTIONS,
            "source_artifact_hashes": hashes,
            "row_count": totals["rounds"],
            "metrics": rates,
            "section_details": details[section],
        }
        if section == "inventory_and_integrity":
            payload["inventory"] = inventory
        _write_json(path, payload)
        paths[section] = path
        records[section] = EdaSection(
            section in _TRUTH_SECTIONS, hashes, int(totals["rounds"]), path
        )
    render_data_card(output_root, profile_hash=profile_hash, inventory=inventory)
    render_validation_report(
        output_root, profile_hash=profile_hash, inventory=inventory, gate_results=gate_results
    )
    return EdaIndex(
        DATASET_EDA_SECTIONS,
        _TRUTH_SECTIONS,
        MappingProxyType(paths),
        chunk_rounds,
        MappingProxyType(records),
    )


def display_eda(index: EdaIndex) -> EdaIndex:
    """Notebook presentation helper; all analysis remains in ``build_dataset_eda``."""
    inventory = json.loads(
        index.output_paths["inventory_and_integrity"].read_text(encoding="utf-8")
    )
    print(
        f"PILOT / NOT FINAL — dataset-profile/config hash "
        f"{inventory['dataset_profile_hash']}. Final claims require the deferred full production dataset."
    )
    return index
