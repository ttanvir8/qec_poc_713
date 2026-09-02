"""Chunked, pilot-only dataset reporting.

The reporting lane is deliberately offline: it reads labels only while making
pre-model diagnostics and never exposes them through observable loaders.
"""

from __future__ import annotations

import json
import struct
import zipfile
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
        pairs.append(
            _Pair(
                condition_id,
                trajectory_id,
                split,
                observable,
                labels,
                (observable_hash, label_hash),
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
    fields = ("detector_bits_packed", "detector_valid_packed", "global_round", "block")
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


def render_data_card(
    output_root: Path, *, profile_hash: str, inventory: Mapping[str, object]
) -> Path:
    path = output_root / "data_card.md"
    path.write_text(
        "# CausalDEM-QEC pilot data card\n\n"
        f"{_PILOT_NOTICE}\n\n"
        f"Dataset profile/config hash: `{profile_hash}`. Package hash: "
        f"`{sha256(Path(__file__).read_bytes()).hexdigest()}`.\n\n"
        "Generation law: round-varying simulator trajectories with exact-XOR canonical classes. "
        "Physical errors, detector observations, and offline labels remain separated.\n\n"
        "Circuits, trajectory/episode/block sizes, physical-error bounds, and split counts are "
        "recorded in the manifest. Intended use is pipeline validation only; forbidden leakage includes labels, sealed seeds, "
        "and cross-partition normalizers. Independent replicates are trajectory IDs.\n\n"
        "Schemas and package/config hashes are recorded with artifacts. Known limitations: this is a "
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
    splits = {
        split: sum(pair.split == split for pair in pairs)
        for split in sorted({pair.split for pair in pairs})
    }
    inventory: dict[str, object] = {
        "trajectories": len(pairs),
        "circuits": sorted({pair.condition_id.split("__", maxsplit=1)[0] for pair in pairs}),
        "conditions": sorted({pair.condition_id for pair in pairs}),
        "splits": splits,
        "generation": manifest.get("generation", {}),
        "schemas": ["observable-v1", "labels-v1"],
        "disk_bytes": sum(
            file.stat().st_size
            for pair in pairs
            for file in (pair.observable / "arrays.npz", pair.labels / "arrays.npz")
        ),
        "failed_jobs": 0,
        "duplicate_ids": 0,
        "profile_hash": profile_hash,
    }
    totals = {
        "rounds": 0,
        "valid": 0,
        "detectors": 0,
        "bits": 0,
        "component": 0.0,
        "classes": 0.0,
        "parity": 0.0,
        "theory": 0.0,
        "queried": 0,
        "block_theory": 0.0,
        "cov_pos": 0,
        "cov_neg": 0,
    }
    sampled: list[np.ndarray] = []
    spectra: list[float] = []
    pacfs: list[float] = []
    acfs: list[float] = []
    block_class_sums: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
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
            selected = class_probability[values["block"][queried] == block]
            key = (pair.condition_id, int(block))
            previous, count = block_class_sums.get(key, (np.zeros(selected.shape[1]), 0))
            block_class_sums[key] = (previous + selected.sum(axis=0), count + len(selected))
        totals["rounds"] += len(values["global_round"])
        totals["valid"] += int(valid.sum())
        totals["detectors"] += valid.size
        totals["bits"] += int(bits.sum())
        totals["component"] += float(values["component_probability"].sum())
        totals["classes"] += float(values["class_probability"].sum())
        totals["parity"] += float(parity.sum())
        totals["theory"] += float(theory.sum())
        totals["queried"] += len(parity)
        covariance = np.cov(bits.astype(float), rowvar=False) if len(bits) > 1 else np.zeros((2, 2))
        totals["cov_pos"] += int((covariance > 0).sum())
        totals["cov_neg"] += int((covariance < 0).sum())
        if len(parity) > 3:
            _, power = signal.periodogram(parity)
            spectra.append(float(np.mean(power)))
            pacf = _pacf(parity)
            pacfs.extend(pacf[1:])
            centered = parity - float(np.mean(parity))
            denominator = float(np.dot(centered, centered))
            if denominator:
                acfs.extend(
                    float(np.dot(centered[:-lag], centered[lag:]) / denominator)
                    for lag in range(1, min(9, len(centered)))
                )
        token = int.from_bytes(
            sha256(f"{sample_seed}:{pair.condition_id}:{pair.trajectory_id}".encode()).digest()[:8],
            "big",
        )
        if (token % max(len(pairs), 1) == 0 or not sampled) and len(parity):
            sampled.append(parity)
    rounds = max(int(totals["rounds"]), 1)
    totals["block_theory"] = sum(
        float(0.5 * (1.0 - np.prod(1.0 - 2.0 * (values / count))) * count)
        for values, count in block_class_sums.values()
    )
    rates = {
        "detector_rate": totals["bits"] / max(int(totals["detectors"]), 1),
        "valid_rate": totals["valid"] / max(int(totals["detectors"]), 1),
        "mean_component_probability": totals["component"] / rounds,
        "mean_class_probability": totals["classes"] / rounds,
        "empirical_parity": totals["parity"] / max(int(totals["queried"]), 1),
        "theoretical_parity": totals["theory"] / max(int(totals["queried"]), 1),
        "block_average_theoretical_parity": totals["block_theory"] / max(int(totals["queried"]), 1),
        "mean_spectral_power": float(np.mean(spectra)) if spectra else 0.0,
        "mean_acf": float(np.mean(acfs)) if acfs else 0.0,
        "mean_pacf": float(np.mean(pacfs)) if pacfs else 0.0,
        "positive_covariances": totals["cov_pos"],
        "negative_covariances": totals["cov_neg"],
    }
    _figure(output_root, profile_hash, np.concatenate(sampled))
    details: dict[str, Mapping[str, object]] = {
        "inventory_and_integrity": inventory,
        "detector_and_logical_rates": {
            "detector_rate": rates["detector_rate"],
            "valid_rate": rates["valid_rate"],
        },
        "physical_and_class_probabilities": {
            "mean_component_probability": rates["mean_component_probability"],
            "mean_class_probability": rates["mean_class_probability"],
        },
        "trajectory_views": {"figure": "trajectory_views.png", "selection": "sha256 deterministic"},
        "temporal_spectra": {
            "mean_acf": rates["mean_acf"],
            "mean_pacf": rates["mean_pacf"],
            "mean_spectral_power": rates["mean_spectral_power"],
        },
        "spatial_correlations": {
            "positive_covariances": rates["positive_covariances"],
            "negative_covariances": rates["negative_covariances"],
        },
        "observation_corruption": {"missing_rate": 1.0 - rates["valid_rate"]},
        "parity_theory_check": {
            "empirical": rates["empirical_parity"],
            "theoretical": rates["theoretical_parity"],
        },
        "noncommutation_gap": {
            "round_mean_minus_block_average": rates["theoretical_parity"]
            - rates["block_average_theoretical_parity"]
        },
        "canonical_catalog": {
            "available_singular_spectra": False,
            "class_mean": rates["mean_class_probability"],
        },
        "split_isolation": {"split_counts": splits, "artifact_hashes": hashes},
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
