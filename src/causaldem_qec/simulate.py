from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
import stim  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from causaldem_qec.artifacts import (
    ArtifactConflict,
    canonical_digest,
    export_checkpoint,
    publish_trajectory,
    verify_trajectory_pair,
    write_manifest,
)
from causaldem_qec.core import (
    CanonicalCatalog,
    CanonicalClass,
    CanonicalDemTruth,
    CircuitSpec,
    ExecutionOptions,
    FailureCode,
    GenerationRequest,
    LabelTrajectory,
    ManifestProvenance,
    ObservableTrajectory,
    PocSpec,
    RunManifest,
    TrajectoryJob,
    TrajectoryResult,
    derive_seed,
    deserialize_manifest_provenance,
    expand_jobs,
    serialize_manifest_provenance,
)

STANDARD_GENERATION_LAW_VERSION = "standard_monolithic_v1"


@dataclass(frozen=True, slots=True)
class PhysicalComponent:
    component_id: int
    kind: str
    targets: tuple[int, ...]
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PhysicalNoisePath:
    component_probability: np.ndarray
    latent_factor: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    missingness_parameters: Mapping[str, float]
    observation_flip_probability: float
    contamination_is_post_sampling: bool
    generator_metadata: Mapping[str, object]


class InvalidPhysicalPath(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BuiltEpisode:
    circuit: stim.Circuit
    detector_round: np.ndarray
    detector_role: np.ndarray
    detector_phase: np.ndarray
    round_instruction_ranges: tuple[tuple[int, int], ...]
    circuit_hash: str


class InvalidCircuit(ValueError):
    pass


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    status: GateStatus
    evidence: Mapping[str, float | int | str | bool]
    affected_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditContext:
    reproducible_hashes: bool
    circuit_dem_valid: bool
    stationary_rate_difference: float
    stationary_rate_tolerance: float
    physical_probability_valid: bool
    episode_indices_isolated: bool
    duplicate_composition_exact: bool
    ambiguity_and_hyperedge_reported: bool
    target_identifiable: bool | None
    observation_budget_difference: float
    observation_budget_tolerance: float
    codrift_expected_sign: int
    codrift_observed_covariance: float
    pre_onset_auc: float
    pre_onset_monte_carlo_half_width: float
    loaders_and_splits_isolated: bool
    stationary_circuit: stim.Circuit | None = None
    stationary_shots: int = 0
    audit_seed: int = 0
    observation_present: np.ndarray | None = None
    observation_valid: np.ndarray | None = None
    observation_mcar_mask: np.ndarray | None = None
    observation_flip_mask: np.ndarray | None = None
    observation_expected_mcar: float | None = None
    observation_burst_mask: np.ndarray | None = None
    observation_expected_burst: float | None = None
    observation_expected_flip: float | None = None
    observation_rate_tolerance: float | None = None
    codrift_samples: np.ndarray | None = None
    codrift_marginal_tolerance: float | None = None
    pre_onset_scores: np.ndarray | None = None
    pre_onset_feature_scores: np.ndarray | None = None
    pre_onset_labels: np.ndarray | None = None


DATASET_GATES = (
    ("DQ01", "reproducible_hashes"),
    ("DQ02", "valid_circuit_dem"),
    ("DQ03", "stationary_monte_carlo"),
    ("DQ04", "physical_probability_bounds"),
    ("DQ05", "episode_index_isolation"),
    ("DQ06", "exact_duplicate_composition"),
    ("DQ07", "ambiguity_and_hyperedge_reported"),
    ("DQ08", "target_identifiability"),
    ("DQ09", "observation_corruption_budget"),
    ("DQ10", "codrift_covariance_sign"),
    ("DQ11", "burst_pre_onset_independence"),
    ("DQ12", "loader_source_split_isolation"),
)


@dataclass(frozen=True, slots=True)
class SampledTrajectory:
    circuit: stim.Circuit
    detector_bits: np.ndarray
    detector_valid: np.ndarray
    logical_observable: np.ndarray
    episode: np.ndarray
    round_in_episode: np.ndarray
    detector_role: np.ndarray
    circuit_phase: np.ndarray


NOISE_KIND = {
    "one_qubit_clifford": ("DEPOLARIZE1", "surface_1q"),
    "two_qubit_clifford": ("DEPOLARIZE2", "surface_2q"),
    "reset_z_basis": ("X_ERROR", "surface_reset"),
    "reset_x_basis": ("Z_ERROR", "surface_reset"),
    "measure_z_basis": ("X_ERROR", "surface_measure"),
    "measure_x_basis": ("Z_ERROR", "surface_measure"),
    "correlated_pair": ("CORRELATED_ERROR", "surface_correlated"),
    "surface_data_idle": ("DEPOLARIZE1", "surface_1q"),
    "repetition_data": ("DEPOLARIZE1", "repetition_data"),
    "repetition_measure": ("X_ERROR", "repetition_measure"),
}


def stationary_ar1(phi: float, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    innovation_sd = np.sqrt(1.0 - phi * phi)
    values = np.empty(size, dtype=np.float64)
    values[0] = rng.normal(size=size[1])
    for index in range(1, size[0]):
        values[index] = phi * values[index - 1] + innovation_sd * rng.normal(size=size[1])
    return values


def bounded_probability(latent: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if np.any(lower <= 0.0) or np.any(upper >= 0.5) or np.any(lower >= upper):
        raise InvalidPhysicalPath("invalid physical probability bounds")
    probability = np.asarray(lower + (upper - lower) * expit(latent), dtype=np.float64)
    if not np.isfinite(probability).all():
        raise InvalidPhysicalPath("nonfinite physical path")
    if np.any(probability <= lower) or np.any(probability >= upper):
        raise InvalidPhysicalPath("saturated bounded transform")
    return probability


def xor_compose(probabilities: np.ndarray) -> float:
    """Return the probability that an odd number of independent faults occurs."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values >= 0.5)):
        raise ValueError("invalid duplicate probabilities")
    return float(-0.5 * np.expm1(np.log1p(-2.0 * values).sum()))


def _dem_errors(
    dem: stim.DetectorErrorModel,
) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    errors: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        detectors: list[int] = []
        logicals: list[int] = []
        for target in instruction.targets_copy():
            if target.is_separator():
                if detectors or logicals:
                    errors.append((probability, tuple(detectors), tuple(logicals)))
                    detectors, logicals = [], []
            elif target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                logicals.append(target.val)
            else:
                raise InvalidCircuit("unsupported DEM target")
        if detectors or logicals:
            errors.append((probability, tuple(detectors), tuple(logicals)))
    return tuple(errors)


def _catalog_from_events(
    events: Sequence[tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...], bool]],
) -> CanonicalCatalog:
    groups: dict[tuple[tuple[int, int, int], ...], list[tuple[float, tuple[int, ...], bool]]] = {}
    for probability, detectors, logical_signature, supported in events:
        groups.setdefault(detectors, []).append((probability, logical_signature, supported))
    classes: list[CanonicalClass] = []
    for class_id, signature in enumerate(sorted(groups)):
        members = groups[signature]
        unique_logicals: set[tuple[int, ...]] = set()
        for _, logical_signature, _ in members:
            unique_logicals.add(logical_signature)
        sorted_logicals = list(unique_logicals)
        sorted_logicals.sort()
        logical_signatures: tuple[tuple[int, ...], ...] = tuple(sorted_logicals)
        probability = xor_compose(np.asarray([member[0] for member in members], dtype=np.float64))
        graphlike = len(signature) in (1, 2)
        supported = all(member[2] for member in members)
        decoder_compatible = len(logical_signatures) == 1 and supported
        classes.append(
            CanonicalClass(
                class_id=class_id,
                detector_signature=signature,
                logical_signatures=logical_signatures,
                probability=probability,
                support_size=len(signature),
                graphlike=graphlike,
                supported=supported,
                decoder_compatible=decoder_compatible,
                adaptable=graphlike and decoder_compatible,
            )
        )
    canonical = [
        {
            "class_id": item.class_id,
            "detectors": item.detector_signature,
            "logicals": item.logical_signatures,
            "probability": item.probability,
            "support_size": item.support_size,
            "graphlike": item.graphlike,
            "supported": item.supported,
            "decoder_compatible": item.decoder_compatible,
            "adaptable": item.adaptable,
        }
        for item in classes
    ]
    graphlike_mass = sum(item.probability for item in classes if item.graphlike)
    adaptable_mass = sum(item.probability for item in classes if item.adaptable)
    ambiguous_mass = sum(item.probability for item in classes if len(item.logical_signatures) != 1)
    hyperedge_mass = sum(item.probability for item in classes if not item.graphlike)
    unsupported_static_mass = sum(
        xor_compose(np.asarray([member[0] for member in members], dtype=np.float64))
        for members in groups.values()
        if not all(member[2] for member in members)
    )
    return CanonicalCatalog(
        classes=tuple(classes),
        duplicate_sizes=tuple(len(groups[item.detector_signature]) for item in classes),
        graphlike_mass=graphlike_mass,
        adaptable_mass=adaptable_mass,
        ambiguous_logical_mass=ambiguous_mass,
        hyperedge_mass=hyperedge_mass,
        unsupported_static_mass=unsupported_static_mass,
        catalog_hash=sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest(),
    )


def canonicalize_test_dem(
    dem: stim.DetectorErrorModel, detector_round: Mapping[int, int]
) -> CanonicalCatalog:
    """Canonicalize a hand-authored DEM fixture without claiming physical source timing."""
    events = []
    for probability, detectors, logicals in _dem_errors(dem):
        if not detectors:
            raise InvalidCircuit("DEM error has no detector support")
        try:
            source_round = min(detector_round[detector] for detector in detectors)
        except KeyError as error:
            raise InvalidCircuit("fixture detector has no round") from error
        signature = tuple(
            sorted((detector_round[detector] - source_round, detector, 0) for detector in detectors)
        )
        events.append((probability, signature, logicals, True))
    return _catalog_from_events(events)


def _episode_layout(
    circuit: stim.Circuit, episodes: Sequence[BuiltEpisode]
) -> tuple[dict[int, tuple[int, int, int, int]], tuple[tuple[int, int, BuiltEpisode], ...]]:
    detectors: dict[int, tuple[int, int, int, int]] = {}
    spans: list[tuple[int, int, BuiltEpisode]] = []
    detector_offset = 0
    instruction_offset = 0
    for episode_index, episode in enumerate(episodes):
        spans.append((instruction_offset, instruction_offset + len(list(episode.circuit)), episode))
        for local, source_round in enumerate(episode.detector_round):
            detectors[detector_offset + local] = (
                episode_index,
                int(source_round),
                int(episode.detector_role[local]),
                int(episode.detector_phase[local]),
            )
        detector_offset += episode.detector_round.size
        instruction_offset += len(list(episode.circuit))
    if detector_offset != circuit.num_detectors:
        raise InvalidCircuit("episode map does not cover trajectory detector IDs")
    return detectors, tuple(spans)


def _canonical_dem_events(
    circuit: stim.Circuit, episode_map: Sequence[BuiltEpisode]
) -> tuple[tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...], bool, int], ...]:
    """Compile and canonicalize an undecomposed trajectory DEM using physical source locations."""
    dem = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=False,
        flatten_loops=True,
    )
    detector_map, spans = _episode_layout(circuit, episode_map)
    explanation_sources: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for explanation in circuit.explain_detector_error_model_errors():
        detector_targets = tuple(
            term.dem_target.val
            for term in explanation.dem_error_terms
            if term.dem_target.is_relative_detector_id()
        )
        logical_targets = tuple(
            term.dem_target.val
            for term in explanation.dem_error_terms
            if term.dem_target.is_logical_observable_id()
        )
        if explanation.circuit_error_locations:
            frames = explanation.circuit_error_locations[0].stack_frames
            if frames:
                explanation_sources.setdefault((detector_targets, logical_targets), []).append(
                    frames[-1].instruction_offset
                )
    events: list[tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...], bool, int]] = []
    for probability, detector_ids, observable_ids in _dem_errors(dem):
        if not detector_ids:
            raise InvalidCircuit("DEM error has no detector support")
        locations = explanation_sources.get((detector_ids, observable_ids), [])
        source_instruction = locations.pop(0) if locations else None
        source_episode = None
        source_round = None
        if source_instruction is not None:
            for candidate_episode, (start, end, episode) in enumerate(spans):
                if start <= source_instruction < end:
                    source_episode = candidate_episode
                    for round_index, (round_start, round_end) in enumerate(
                        episode.round_instruction_ranges
                    ):
                        if round_start <= source_instruction - start < round_end:
                            source_round = round_index
                            break
                    break
        detector_episodes = {detector_map[item][0] for item in detector_ids}
        observable_episodes = set(observable_ids)
        if len(detector_episodes) != 1 or (
            observable_episodes and observable_episodes != detector_episodes
        ):
            raise InvalidCircuit("DEM event crosses an episode boundary")
        episode_index = next(iter(detector_episodes))
        if source_episode != episode_index or source_round is None:
            signature = tuple(
                sorted(
                    (detector_map[item][1], detector_map[item][2], detector_map[item][3])
                    for item in detector_ids
                )
            )
            events.append((probability, signature, (), False, -1))
            continue
        signature = tuple(
            sorted(
                (detector_map[item][1] - source_round, detector_map[item][2], detector_map[item][3])
                for item in detector_ids
            )
        )
        logical = tuple(sorted(item - episode_index for item in observable_ids))
        events.append((probability, signature, logical, True, episode_index * 32 + source_round))
    return tuple(events)


def canonicalize_dem(
    circuit: stim.Circuit, episode_map: Sequence[BuiltEpisode]
) -> CanonicalCatalog:
    events = _canonical_dem_events(circuit, episode_map)
    return _catalog_from_events(tuple(event[:4] for event in events))


def canonicalize_dem_truth(
    circuit: stim.Circuit, episode_map: Sequence[BuiltEpisode]
) -> CanonicalDemTruth:
    """Derive frozen round truth from the complete, undecomposed trajectory DEM."""
    events = _canonical_dem_events(circuit, episode_map)
    catalog = _catalog_from_events(tuple(event[:4] for event in events))
    class_index = {item.detector_signature: item.class_id for item in catalog.classes}
    probability = np.zeros((len(episode_map) * 32, len(catalog.classes)), dtype=np.float64)
    by_round_class: dict[tuple[int, int], list[float]] = {}
    for event_probability, signature, _, _, source_round in events:
        if source_round >= 0:
            by_round_class.setdefault((source_round, class_index[signature]), []).append(
                event_probability
            )
    for (round_index, class_id), values in by_round_class.items():
        probability[round_index, class_id] = xor_compose(np.asarray(values, dtype=np.float64))
    dem = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=False,
        flatten_loops=True,
    )
    return CanonicalDemTruth(
        catalog=catalog,
        class_probability=probability,
        dem_hash=sha256(str(dem).encode("utf-8")).hexdigest(),
    )


def run_dataset_gates(context: AuditContext) -> tuple[GateResult, ...]:
    """Evaluate all dataset gates and retain evidence even when a gate is incomplete."""
    simple = (
        ("DQ01", context.reproducible_hashes, {"reproducible_hashes": context.reproducible_hashes}),
        ("DQ02", context.circuit_dem_valid, {"circuit_dem_valid": context.circuit_dem_valid}),
        (
            "DQ04",
            context.physical_probability_valid,
            {"physical_probability_valid": context.physical_probability_valid},
        ),
        (
            "DQ05",
            context.episode_indices_isolated,
            {"episode_indices_isolated": context.episode_indices_isolated},
        ),
        (
            "DQ06",
            context.duplicate_composition_exact,
            {"duplicate_composition_exact": context.duplicate_composition_exact},
        ),
        (
            "DQ07",
            context.ambiguity_and_hyperedge_reported,
            {"ambiguity_and_hyperedge_reported": context.ambiguity_and_hyperedge_reported},
        ),
        (
            "DQ12",
            context.loaders_and_splits_isolated,
            {"loaders_and_splits_isolated": context.loaders_and_splits_isolated},
        ),
    )
    results = [
        GateResult(
            gate_id, GateStatus.PASS if passed else GateStatus.FAIL, MappingProxyType(evidence), ()
        )
        for gate_id, passed, evidence in simple
    ]
    stationary_difference = context.stationary_rate_difference
    stationary_tolerance = context.stationary_rate_tolerance
    stationary_evidence: dict[str, float | int | str | bool] = {
        "difference": stationary_difference,
        "tolerance": stationary_tolerance,
    }
    if context.stationary_circuit is not None and context.stationary_shots > 0:
        sampler = context.stationary_circuit.compile_detector_sampler(seed=context.audit_seed)
        detectors, logicals = sampler.sample(
            shots=context.stationary_shots, separate_observables=True
        )
        detector_rate = float(np.mean(detectors, dtype=np.float64))
        logical_rate = float(np.mean(logicals, dtype=np.float64))
        stationary_difference = abs(detector_rate - logical_rate)
        stationary_tolerance = (
            max(
                3.0 * np.sqrt(detector_rate * (1.0 - detector_rate) / context.stationary_shots),
                3.0 * np.sqrt(logical_rate * (1.0 - logical_rate) / context.stationary_shots),
            )
            + 1.0 / context.stationary_shots
        )
        stationary_evidence.update(
            detector_rate=detector_rate,
            logical_rate=logical_rate,
            shots=context.stationary_shots,
            difference=stationary_difference,
            tolerance=stationary_tolerance,
        )
    observation_difference = context.observation_budget_difference
    observation_tolerance = context.observation_budget_tolerance
    observation_evidence: dict[str, float | int | str | bool] = {
        "difference": observation_difference,
        "tolerance": observation_tolerance,
    }
    if context.observation_present is not None and context.observation_valid is not None:
        present = context.observation_present
        valid = context.observation_valid
        if present.shape != valid.shape or present.dtype != np.bool_ or valid.dtype != np.bool_:
            raise ValueError("invalid observation audit arrays")
        denominator = int(present.sum())
        if denominator == 0:
            raise ValueError("observation audit has no present values")
        burst = context.observation_burst_mask
        mcar = context.observation_mcar_mask
        if burst is not None:
            if burst.shape != present.shape or burst.dtype != np.bool_:
                raise ValueError("invalid burst audit array")
            if mcar is None:
                raise ValueError("burst audit requires an independent MCAR mask")
        if mcar is None:
            mcar = present & ~valid
        if mcar.shape != present.shape or mcar.dtype != np.bool_:
            raise ValueError("invalid MCAR audit array")
        combined_missing = mcar if burst is None else mcar | burst
        if not np.array_equal(valid, present & ~combined_missing):
            raise ValueError("observation validity does not match missingness masks")
        mcar_observed = present & mcar
        if burst is not None:
            mcar_observed &= ~burst
        mcar_rate = float(np.count_nonzero(mcar_observed) / denominator)
        expected = (
            context.observation_expected_mcar
            if context.observation_expected_mcar is not None
            else 0.0
        )
        tolerance = context.observation_rate_tolerance
        mcar_tolerance = (
            tolerance
            if tolerance is not None
            else float(3.0 * np.sqrt(expected * (1.0 - expected) / denominator) + 1.0 / denominator)
        )
        differences = [abs(mcar_rate - expected)]
        observation_evidence.update(
            mcar_rate=mcar_rate,
            expected_mcar=expected,
            mcar_tolerance=mcar_tolerance,
            samples=denominator,
        )
        if burst is not None:
            burst_rate = float(np.count_nonzero(burst & present & ~valid) / denominator)
            burst_expected = (
                context.observation_expected_burst
                if context.observation_expected_burst is not None
                else 0.0
            )
            burst_tolerance = (
                tolerance
                if tolerance is not None
                else float(
                    3.0 * np.sqrt(burst_expected * (1.0 - burst_expected) / denominator)
                    + 1.0 / denominator
                )
            )
            differences.append(abs(burst_rate - burst_expected))
            observation_evidence.update(
                burst_rate=burst_rate,
                expected_burst=burst_expected,
                burst_tolerance=burst_tolerance,
            )
            if abs(burst_rate - burst_expected) > burst_tolerance:
                differences.append(float("inf"))
        if context.observation_flip_mask is not None:
            if context.observation_flip_mask.shape != present.shape:
                raise ValueError("invalid flip audit array")
            flip_rate = float(
                np.count_nonzero(context.observation_flip_mask & present) / denominator
            )
            flip_expected = (
                context.observation_expected_flip
                if context.observation_expected_flip is not None
                else 0.0
            )
            flip_tolerance = (
                tolerance
                if tolerance is not None
                else float(
                    3.0 * np.sqrt(flip_expected * (1.0 - flip_expected) / denominator)
                    + 1.0 / denominator
                )
            )
            differences.append(abs(flip_rate - flip_expected))
            observation_evidence.update(
                flip_rate=flip_rate, expected_flip=flip_expected, flip_tolerance=flip_tolerance
            )
            if abs(flip_rate - flip_expected) > flip_tolerance:
                differences.append(float("inf"))
        observation_difference = max(differences)
        observation_tolerance = mcar_tolerance
        observation_evidence.update(
            difference=observation_difference, tolerance=observation_tolerance
        )
    covariance = context.codrift_observed_covariance
    marginal_difference = 0.0
    if context.codrift_samples is not None:
        samples = context.codrift_samples
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("codrift samples must have two columns")
        if context.codrift_marginal_tolerance is None:
            raise ValueError("codrift samples require codrift_marginal_tolerance")
        covariance = float(np.cov(samples[:, 0], samples[:, 1], bias=True)[0, 1])
        marginal_difference = abs(float(np.ptp(samples[:, 0])) - float(np.ptp(samples[:, 1])))
    codrift_evidence: dict[str, float | int | str | bool] = {
        "expected_sign": context.codrift_expected_sign,
        "observed_covariance": covariance,
        "marginal_range_difference": marginal_difference,
    }
    codrift_tolerance = (
        context.codrift_marginal_tolerance
        if context.codrift_marginal_tolerance is not None
        else context.observation_budget_tolerance
    )
    codrift_evidence["marginal_tolerance"] = codrift_tolerance
    auc = context.pre_onset_auc
    threshold = context.pre_onset_monte_carlo_half_width
    if context.pre_onset_labels is not None:
        scores = context.pre_onset_feature_scores
        if scores is None and context.pre_onset_scores is not None:
            scores = context.pre_onset_scores[:, None]
        if scores is not None:
            if scores.ndim != 2 or scores.shape[0] != context.pre_onset_labels.size:
                raise ValueError("invalid pre-onset feature scores")
            aucs = np.asarray(
                [
                    mann_whitney_auc(scores[:, column], context.pre_onset_labels)
                    for column in range(scores.shape[1])
                ]
            )
            auc = float(aucs[np.argmax(np.abs(aucs - 0.5))])
            rng = np.random.default_rng(context.audit_seed)
            deviations = np.empty(256, dtype=np.float64)
            for index in range(deviations.size):
                permuted = rng.permutation(context.pre_onset_labels)
                deviations[index] = max(
                    abs(mann_whitney_auc(scores[:, column], permuted) - 0.5)
                    for column in range(scores.shape[1])
                )
            threshold = float(np.quantile(deviations, 0.99, method="higher"))
    onset_evidence: dict[str, float | int | str | bool] = {
        "auc": auc,
        "max_abs_departure": abs(auc - 0.5),
        "permutation_threshold": threshold,
    }
    results.extend(
        (
            GateResult(
                "DQ03",
                GateStatus.PASS
                if stationary_difference <= stationary_tolerance
                else GateStatus.FAIL,
                MappingProxyType(stationary_evidence),
                (),
            ),
            GateResult(
                "DQ08", GateStatus.NOT_RUN, MappingProxyType({"target_identifiable": "not_run"}), ()
            ),
            GateResult(
                "DQ09",
                GateStatus.PASS
                if observation_difference <= observation_tolerance
                else GateStatus.FAIL,
                MappingProxyType(observation_evidence),
                (),
            ),
            GateResult(
                "DQ10",
                GateStatus.PASS
                if context.codrift_expected_sign * covariance > 0.0
                and marginal_difference <= codrift_tolerance
                else GateStatus.FAIL,
                MappingProxyType(codrift_evidence),
                (),
            ),
            GateResult(
                "DQ11",
                GateStatus.PASS if abs(auc - 0.5) <= threshold else GateStatus.FAIL,
                MappingProxyType(onset_evidence),
                (),
            ),
        )
    )
    by_id = {item.gate_id: item for item in results}
    return tuple(by_id[gate_id] for gate_id, _ in DATASET_GATES)


def dataset_gates_complete(results: Sequence[GateResult]) -> bool:
    """A verification gate is complete only when every registered audit passed."""
    return len(results) == len(DATASET_GATES) and all(
        item.status is GateStatus.PASS for item in results
    )


def stationary_stim_audit(circuit: stim.Circuit, shots: int, seed: int) -> tuple[float, float]:
    """Return direct Stim detector rate and its prescribed binomial tolerance."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    samples = circuit.compile_detector_sampler(seed=seed).sample(shots=shots)
    rate = float(np.mean(samples, dtype=np.float64))
    return rate, float(3.0 * np.sqrt(rate * (1.0 - rate) / shots) + 1.0 / shots)


def mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute a tie-aware rank AUC without a statistics-framework dependency."""
    values = np.asarray(scores, dtype=np.float64)
    onset = np.asarray(labels, dtype=np.bool_)
    if values.ndim != 1 or onset.shape != values.shape or not onset.any() or onset.all():
        raise ValueError("invalid onset score inputs")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end + 1.0) / 2.0
        cursor = end
    positives = int(onset.sum())
    negatives = values.size - positives
    return float((ranks[onset].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def onset_permutation_threshold(scores: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Return the deterministic 99th-percentile null deviation from 256 permutations."""
    rng = np.random.default_rng(seed)
    onset = np.asarray(labels, dtype=np.bool_)
    deviations = np.empty(256, dtype=np.float64)
    for index in range(deviations.size):
        deviations[index] = abs(mann_whitney_auc(scores, rng.permutation(onset)) - 0.5)
    return float(np.quantile(deviations, 0.99, method="higher"))


def _template(circuit: CircuitSpec) -> stim.Circuit:
    task = (
        "repetition_code:memory"
        if circuit.family == "repetition"
        else "surface_code:rotated_memory_z"
    )
    return stim.Circuit.generated(task, distance=circuit.distance, rounds=32).flattened()


def _target_values(targets: list[stim.GateTarget]) -> tuple[int, ...]:
    return tuple(target.qubit_value for target in targets)


def _component_operations(
    circuit: CircuitSpec, instruction: stim.CircuitInstruction
) -> tuple[str, ...]:
    name = instruction.name
    if circuit.family == "repetition":
        if name == "CX":
            return ("repetition_data",)
        if name in {"MR", "M"}:
            return ("repetition_measure",)
        return ()
    if name == "H":
        return ("one_qubit_clifford",)
    if name == "CX":
        return ("two_qubit_clifford", "correlated_pair")
    if name == "R":
        return ("reset_z_basis",)
    if name == "MR":
        return ("measure_z_basis", "reset_z_basis")
    if name == "M":
        return ("measure_z_basis",)
    return ()


def _data_qubits(template: stim.Circuit) -> frozenset[int]:
    values = {
        target.qubit_value
        for instruction in template
        if instruction.name == "M"
        for target in instruction.targets_copy()
    }
    if not values:
        raise InvalidCircuit("memory template has no final data measurement")
    return frozenset(values)


def _component_sites(circuit: CircuitSpec) -> tuple[tuple[str, tuple[int, ...]], ...]:
    template = _template(circuit)
    data_qubits = _data_qubits(template)
    active_data: set[int] = set()
    sites: list[tuple[str, tuple[int, ...]]] = []
    for instruction in template:
        targets = instruction.targets_copy()
        target_values = _target_values(targets)
        if circuit.family == "surface" and instruction.name == "TICK":
            idle = tuple(sorted(data_qubits - active_data))
            if idle:
                sites.append(("surface_data_idle", idle))
            active_data.clear()
            continue
        for kind in _component_operations(circuit, instruction):
            if circuit.family == "repetition" and kind == "repetition_data":
                target_values = tuple(target for target in target_values if target in data_qubits)
            sites.append((kind, target_values))
        if circuit.family == "surface":
            active_data.update(target for target in target_values if target in data_qubits)
    return tuple(sites)


def _layout_bounds(spec: PocSpec | None, bound_kind: str) -> tuple[float, float]:
    if spec is not None:
        return spec.component_bounds[bound_kind]
    # The public layout has no PocSpec argument; generation always supplies it.
    return {
        "repetition_data": (0.0001, 0.03),
        "repetition_measure": (0.0001, 0.05),
        "surface_1q": (0.00001, 0.01),
        "surface_2q": (0.00001, 0.02),
        "surface_reset": (0.00001, 0.02),
        "surface_measure": (0.00001, 0.03),
        "surface_correlated": (0.000001, 0.005),
    }[bound_kind]


def component_layout(
    circuit: CircuitSpec, spec: PocSpec | None = None
) -> tuple[PhysicalComponent, ...]:
    """Fix each circuit operation location to one heterogeneous physical-rate column."""
    components: list[PhysicalComponent] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for kind, targets in _component_sites(circuit):
        key = kind, targets
        if key in seen:
            continue
        seen.add(key)
        bound_kind = NOISE_KIND[kind][1]
        components.append(
            PhysicalComponent(len(components), kind, targets, *_layout_bounds(spec, bound_kind))
        )
    if not components:
        raise InvalidCircuit(f"no physical components in {circuit.circuit_id}")
    return tuple(components)


def _rng(spec: PocSpec, job: TrajectoryJob, stream: str, attempt: int = 0) -> np.random.Generator:
    schema_version = _integer(spec.raw["schema_version"], "schema_version")
    seed = derive_seed(
        job.root_seed,
        schema_version,
        job.condition_id,
        job.trajectory_id,
        stream,
        attempt,
    )
    return np.random.default_rng(seed)


def _config_float(spec: PocSpec, family: str, name: str) -> float:
    return _number(spec.dynamics[family][name], f"{family}.{name}")


def _config_pair(spec: PocSpec, family: str, name: str) -> tuple[float, float]:
    values = spec.dynamics[family][name]
    if not isinstance(values, tuple) or len(values) != 2:
        raise InvalidPhysicalPath(f"invalid {family}.{name}")
    return _number(values[0], f"{family}.{name}[0]"), _number(values[1], f"{family}.{name}[1]")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidPhysicalPath(f"invalid {name}")
    number = float(value)
    if not np.isfinite(number):
        raise InvalidPhysicalPath(f"nonfinite {name}")
    return number


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidPhysicalPath(f"invalid {name}")
    return value


def _f03_latent(
    spec: PocSpec,
    job: TrajectoryJob,
    total_rounds: int,
    components: tuple[PhysicalComponent, ...],
    attempt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = spec.dynamics["f03"]
    shared_phi = _config_pair(spec, "f03", "shared_phi")
    dynamics_rng = _rng(spec, job, "dynamics", attempt)
    shared = np.column_stack(
        [
            stationary_ar1(shared_phi[0], (total_rounds, 1), dynamics_rng)[:, 0],
            stationary_ar1(shared_phi[1], (total_rounds, 1), dynamics_rng)[:, 0],
        ]
    )
    local = stationary_ar1(
        _number(config["local_phi"], "f03.local_phi"), (total_rounds, len(components)), dynamics_rng
    ) * _number(config["local_sd"], "f03.local_sd")
    type_sign = config["type_sign"]
    if not isinstance(type_sign, Mapping):
        raise InvalidPhysicalPath("invalid f03.type_sign")
    signs = np.asarray(
        [
            _number(type_sign[NOISE_KIND[component.kind][1]], "f03.type_sign")
            for component in components
        ],
        dtype=np.float64,
    )
    geometry = np.linspace(-1.0, 1.0, len(components), dtype=np.float64)
    if len(components) > 1:
        geometry /= np.linalg.norm(geometry)
    loadings = (
        np.column_stack(
            [
                np.full(len(components), _number(config["global_loading"], "f03.global_loading")),
                _number(config["x_loading"], "f03.x_loading") * geometry,
            ]
        )
        * signs[:, None]
    )
    latent = shared @ loadings.T + local
    return latent, shared, loadings


def _geometric_run_mask(
    rounds: int, hazard: float, mean_duration: float, rng: np.random.Generator
) -> np.ndarray:
    mask = np.zeros(rounds, dtype=np.bool_)
    index = 0
    while index < rounds:
        if rng.random() < hazard:
            duration = int(rng.geometric(1.0 / mean_duration))
            mask[index : min(rounds, index + duration)] = True
            index += duration
        else:
            index += 1
    return mask


def _canonical_component_pairs(
    components: tuple[PhysicalComponent, ...],
) -> tuple[tuple[int, int], ...]:
    by_type: dict[str, list[PhysicalComponent]] = {}
    for component in components:
        by_type.setdefault(NOISE_KIND[component.kind][1], []).append(component)
    pairs: list[tuple[int, int]] = []
    for physical_type in sorted(by_type):
        motifs = sorted(by_type[physical_type], key=lambda item: (item.targets, item.component_id))
        if len(motifs) % 2:
            raise InvalidPhysicalPath(f"unmatched f14 component layout for {physical_type}")
        pairs.extend(
            (motifs[index].component_id, motifs[index + 1].component_id)
            for index in range(0, len(motifs), 2)
        )
    return tuple(pairs)


def generate_dynamics(
    spec: PocSpec,
    job: TrajectoryJob,
    scored_rounds: int | None = None,
    burn_in: int | None = None,
    *,
    attempt: int = 0,
) -> PhysicalNoisePath:
    scored = spec.scored_rounds if scored_rounds is None else scored_rounds
    discarded = spec.burn_in_rounds if burn_in is None else burn_in
    if scored <= 0 or discarded < 0:
        raise InvalidPhysicalPath("round counts must be positive after burn-in")
    components = component_layout(job.circuit, spec)
    lower = np.asarray([component.lower for component in components], dtype=np.float64)
    upper = np.asarray([component.upper for component in components], dtype=np.float64)
    total = scored + discarded
    dynamics_rng = _rng(spec, job, "dynamics", attempt)
    match job.dynamics_id:
        case "f01":
            offsets = dynamics_rng.normal(
                scale=_config_float(spec, "f01", "offset_sd"), size=len(components)
            )
            all_latent = np.broadcast_to(offsets, (total, len(components))).copy()
            all_factors = all_latent[:, : min(2, len(components))]
        case "f02":
            all_latent = stationary_ar1(
                _config_float(spec, "f02", "phi"), (total, len(components)), dynamics_rng
            ) * _config_float(spec, "f02", "loading")
            all_factors = all_latent[:, : min(2, len(components))]
        case "f03" | "f07" | "f08" | "f12":
            all_latent, all_factors, loadings = _f03_latent(spec, job, total, components, attempt)
            if job.dynamics_id == "f12":
                config = spec.dynamics["f12"]
                burst = _geometric_run_mask(
                    total,
                    _number(config["onset_hazard"], "f12.onset_hazard"),
                    _number(config["mean_duration"], "f12.mean_duration"),
                    _rng(spec, job, "burst", attempt),
                )
                all_latent += (
                    _number(config["amplitude"], "f12.amplitude")
                    * burst[:, None]
                    * loadings.sum(axis=1)
                )
        case "f06":
            periods = np.linspace(
                _config_float(spec, "f06", "start_period"),
                _config_float(spec, "f06", "stop_period"),
                total,
            )
            phase = dynamics_rng.uniform(0.0, 2.0 * np.pi)
            chirp = np.sin(2.0 * np.pi * np.cumsum(1.0 / periods) + phase)
            all_latent = _config_float(spec, "f06", "amplitude") * chirp[:, None]
            all_factors = np.column_stack(
                (chirp, np.cos(2.0 * np.pi * np.cumsum(1.0 / periods) + phase))
            )
        case "f14_positive" | "f14_negative":
            config = spec.dynamics[job.dynamics_id]
            pairs = _canonical_component_pairs(components)
            pair_count = len(pairs)
            driver = stationary_ar1(
                _number(config["phi"], "f14.phi"), (total, pair_count), dynamics_rng
            )
            sign = _number(config["sign"], "f14.sign")
            all_latent = np.empty((total, len(components)), dtype=np.float64)
            amplitude = _number(config["amplitude"], "f14.amplitude")
            for pair_index, (first, second) in enumerate(pairs):
                all_latent[:, first] = amplitude * driver[:, pair_index]
                all_latent[:, second] = amplitude * sign * driver[:, pair_index]
            all_factors = np.column_stack((driver[:, 0], sign * driver[:, 0]))
        case _:
            raise InvalidPhysicalPath(f"unknown dynamics family {job.dynamics_id}")
    probability = bounded_probability(all_latent[discarded:], lower, upper)
    if probability.dtype != np.dtype(np.float64) or probability.shape != (scored, len(components)):
        raise InvalidPhysicalPath("invalid physical path shape")
    missingness: Mapping[str, float] = MappingProxyType({})
    flip_probability = 0.0
    contamination = False
    if job.dynamics_id == "f07":
        config = spec.dynamics["f07"]
        missingness = MappingProxyType(
            {
                "mcar": _number(config["mcar"], "f07.mcar"),
                "burst_hazard": _number(config["burst_hazard"], "f07.burst_hazard"),
                "mean_duration": _number(config["mean_duration"], "f07.mean_duration"),
                "detector_fraction": _number(config["detector_fraction"], "f07.detector_fraction"),
            }
        )
    if job.dynamics_id == "f08":
        flip_probability = _config_float(spec, "f08", "flip_probability")
        contamination = True
    base_parameters: object = MappingProxyType({})
    base = spec.dynamics[job.dynamics_id].get("base")
    if isinstance(base, str):
        base_parameters = spec.dynamics[base]
    layout_commitment = sha256(
        repr(tuple((item.kind, item.targets) for item in components)).encode("utf-8")
    ).hexdigest()
    metadata: Mapping[str, object] = MappingProxyType(
        {
            "dynamics_id": job.dynamics_id,
            "burn_in_rounds": discarded,
            "burn_in_final_state": tuple(float(value) for value in all_latent[discarded - 1])
            if discarded
            else (),
            "resolved_parameters": spec.dynamics[job.dynamics_id],
            "base_parameters": base_parameters,
            "component_layout_commitment": layout_commitment,
            "component_bounds": tuple(
                (NOISE_KIND[item.kind][1], item.lower, item.upper) for item in components
            ),
            "attempt": attempt,
        }
    )
    return PhysicalNoisePath(
        component_probability=probability,
        latent_factor=np.asarray(all_factors[discarded:], dtype=np.float64),
        lower_bound=lower,
        upper_bound=upper,
        missingness_parameters=missingness,
        observation_flip_probability=flip_probability,
        contamination_is_post_sampling=contamination,
        generator_metadata=metadata,
    )


def append_noise(
    circuit: stim.Circuit, name: str, targets: list[stim.GateTarget], p: float
) -> None:
    if not 0.0 < p < 0.5:
        raise ValueError(f"invalid {name} probability {p}")
    circuit.append(name, targets, p)


def _append_instruction(circuit: stim.Circuit, instruction: stim.CircuitInstruction) -> None:
    circuit.append(instruction.name, instruction.targets_copy(), instruction.gate_args_copy())


def _detector_metadata(circuit: stim.Circuit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds: list[int] = []
    roles: list[int] = []
    phases: list[int] = []
    per_round: dict[int, int] = {}
    for instruction in circuit:
        if instruction.name != "DETECTOR":
            continue
        coordinates = instruction.gate_args_copy()
        if not coordinates:
            raise InvalidCircuit("detector without a time coordinate")
        round_index = min(int(coordinates[-1]), 31)
        role = per_round.get(round_index, 0)
        per_round[round_index] = role + 1
        rounds.append(round_index)
        roles.append(role)
        phases.append(int(coordinates[-1]))
    return (
        np.asarray(rounds, dtype=np.uint8),
        np.asarray(roles, dtype=np.uint16),
        np.asarray(phases, dtype=np.uint8),
    )


def build_memory_episode(
    circuit: CircuitSpec,
    round_probability: np.ndarray,
    episode_id: int,
    spec: PocSpec | None = None,
) -> BuiltEpisode:
    components = component_layout(circuit, spec)
    if round_probability.dtype != np.dtype(np.float64) or round_probability.shape != (
        32,
        len(components),
    ):
        raise InvalidCircuit(
            "round probabilities must be float64 with one row per round and component"
        )
    if not np.isfinite(round_probability).all() or np.any(
        (round_probability <= 0.0) | (round_probability >= 0.5)
    ):
        raise InvalidCircuit("round probabilities must be finite physical probabilities")
    lookup = {
        (component.kind, component.targets): component.component_id for component in components
    }
    result = stim.Circuit()
    current_round = 0
    template = _template(circuit)
    data_qubits = _data_qubits(template)
    active_data: set[int] = set()
    range_start = 0
    round_ranges: list[tuple[int, int]] = []
    for instruction in template:
        targets = instruction.targets_copy()
        target_key = _target_values(targets)
        if circuit.family == "surface" and instruction.name == "TICK":
            _append_instruction(result, instruction)
            idle = tuple(sorted(data_qubits - active_data))
            if idle:
                column = lookup[("surface_data_idle", idle)]
                append_noise(
                    result,
                    NOISE_KIND["surface_data_idle"][0],
                    [stim.GateTarget(target) for target in idle],
                    float(round_probability[min(current_round, 31), column]),
                )
            active_data.clear()
            continue
        kinds = _component_operations(circuit, instruction)
        noise_targets = targets
        noise_target_key = target_key
        if circuit.family == "repetition" and instruction.name == "CX":
            noise_targets = [target for target in targets if target.qubit_value in data_qubits]
            noise_target_key = _target_values(noise_targets)
        if instruction.name in {"MR", "M"}:
            measurement_kind = (
                "repetition_measure" if circuit.family == "repetition" else "measure_z_basis"
            )
            column = lookup[(measurement_kind, target_key)]
            append_noise(
                result,
                NOISE_KIND[measurement_kind][0],
                targets,
                float(round_probability[min(current_round, 31), column]),
            )
        if instruction.name == "OBSERVABLE_INCLUDE":
            result.append("OBSERVABLE_INCLUDE", targets, episode_id)
            continue
        _append_instruction(result, instruction)
        for kind in kinds:
            if (
                circuit.family == "surface"
                and instruction.name in {"MR", "M"}
                and kind == "measure_z_basis"
            ):
                continue
            if circuit.family == "surface" and instruction.name == "MR" and kind == "reset_z_basis":
                pass
            elif (circuit.family == "repetition" and instruction.name in {"MR", "M"}) or not kinds:
                continue
            column = lookup[(kind, noise_target_key)]
            noise_name = NOISE_KIND[kind][0]
            probability = float(round_probability[min(current_round, 31), column])
            if kind == "correlated_pair":
                if len(noise_targets) < 2 or len(noise_targets) % 2:
                    raise InvalidCircuit("correlated pair requires two qubits")
                for pair_start in range(0, len(noise_targets), 2):
                    append_noise(
                        result,
                        noise_name,
                        [
                            stim.target_x(noise_targets[pair_start].qubit_value),
                            stim.target_x(noise_targets[pair_start + 1].qubit_value),
                        ],
                        probability,
                    )
            else:
                append_noise(result, noise_name, noise_targets, probability)
        if circuit.family == "surface":
            active_data.update(target for target in target_key if target in data_qubits)
        if instruction.name == "MR":
            round_ranges.append((range_start, len(list(result))))
            range_start = len(list(result))
            current_round += 1
    if current_round != 32 or "REPEAT" in str(result):
        raise InvalidCircuit("episode expansion did not fully unroll 32 rounds")
    if len(round_ranges) != 32:
        raise InvalidCircuit("episode is missing round instruction ranges")
    round_ranges[-1] = (round_ranges[-1][0], len(list(result)))
    detector_round, detector_role, detector_phase = _detector_metadata(result)
    circuit_hash = sha256(str(result).encode("utf-8")).hexdigest()
    return BuiltEpisode(
        circuit=result,
        detector_round=detector_round,
        detector_role=detector_role,
        detector_phase=detector_phase,
        round_instruction_ranges=tuple(round_ranges),
        circuit_hash=circuit_hash,
    )


def _apply_missingness(
    bits: np.ndarray, present: np.ndarray, parameters: Mapping[str, float], rng: np.random.Generator
) -> np.ndarray:
    valid = present.copy()
    if not parameters:
        return valid
    missing = np.asarray(rng.random(bits.shape) < parameters["mcar"], dtype=np.bool_)
    burst = _geometric_run_mask(
        bits.shape[0], parameters["burst_hazard"], parameters["mean_duration"], rng
    )
    subset = np.asarray(rng.random(bits.shape[1]) < parameters["detector_fraction"], dtype=np.bool_)
    if not subset.any():
        subset[rng.integers(bits.shape[1])] = True
    missing |= burst[:, None] & subset[None, :]
    valid[present & missing] = False
    return valid


def sample_trajectory(
    spec: PocSpec, job: TrajectoryJob, path: PhysicalNoisePath, *, attempt: int = 0
) -> SampledTrajectory:
    rounds = path.component_probability.shape[0]
    if rounds == 0 or rounds % spec.episode_rounds:
        raise InvalidPhysicalPath("sampled rounds must contain complete episodes")
    components = component_layout(job.circuit)
    if path.component_probability.shape[1] != len(components):
        raise InvalidPhysicalPath("physical path component layout mismatch")
    episode_count = rounds // spec.episode_rounds
    built_episodes = [
        build_memory_episode(
            job.circuit,
            path.component_probability[index * 32 : (index + 1) * 32],
            index,
            spec,
        )
        for index in range(episode_count)
    ]
    trajectory_circuit = stim.Circuit()
    for built in built_episodes:
        trajectory_circuit += built.circuit
    seed_sequence = derive_seed(
        job.root_seed,
        _integer(spec.raw["schema_version"], "schema_version"),
        job.condition_id,
        job.trajectory_id,
        "fault",
        attempt,
    )
    fault_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
    sampler = trajectory_circuit.compile_detector_sampler(seed=fault_seed)
    detectors, observables = sampler.sample(shots=1, separate_observables=True, bit_packed=False)
    max_role = max(int(built.detector_role.max()) for built in built_episodes) + 1
    detector_bits = np.zeros((rounds, max_role), dtype=np.bool_)
    present = np.zeros((rounds, max_role), dtype=np.bool_)
    offset = 0
    for episode_index, built in enumerate(built_episodes):
        count = built.detector_round.size
        for local_index in range(count):
            row = episode_index * 32 + int(built.detector_round[local_index])
            column = int(built.detector_role[local_index])
            detector_bits[row, column] = detectors[0, offset + local_index]
            present[row, column] = True
        offset += count
    if offset != detectors.shape[1]:
        raise InvalidCircuit("detector metadata does not cover sampler output")
    detector_valid = _apply_missingness(
        detector_bits, present, path.missingness_parameters, _rng(spec, job, "missingness", attempt)
    )
    if path.contamination_is_post_sampling:
        flips = (
            _rng(spec, job, "contamination", attempt).random(detector_bits.shape)
            < path.observation_flip_probability
        )
        detector_bits[present & flips] ^= True
    global_round = np.arange(rounds, dtype=np.uint32)
    return SampledTrajectory(
        circuit=trajectory_circuit,
        detector_bits=detector_bits,
        detector_valid=detector_valid,
        logical_observable=observables[0].astype(np.bool_, copy=False),
        episode=global_round // np.uint32(32),
        round_in_episode=global_round % np.uint32(32),
        detector_role=np.arange(max_role, dtype=np.uint16),
        circuit_phase=(global_round % np.uint32(32)).astype(np.uint8),
    )


def _stream_commitments(spec: PocSpec, job: TrajectoryJob, attempt: int) -> Mapping[str, str]:
    schema_value = spec.raw["schema_version"]
    if not isinstance(schema_value, int):
        raise InvalidPhysicalPath("invalid schema version")
    schema_version = schema_value
    streams = ("dynamics", "burst", "fault", "missingness", "contamination")
    return MappingProxyType(
        {
            stream: hashlib.sha256(
                derive_seed(
                    job.root_seed,
                    schema_version,
                    job.condition_id,
                    job.trajectory_id,
                    stream,
                    attempt,
                )
                .generate_state(8, dtype=np.uint32)
                .tobytes()
            ).hexdigest()
            for stream in streams
        }
    )


def _package_versions() -> Mapping[str, str]:
    return MappingProxyType(
        {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "stim", "pymatching")
        }
    )


def _future_block_probability(class_probability: np.ndarray, block_rounds: int) -> np.ndarray:
    future = np.zeros_like(class_probability)
    block_count = class_probability.shape[0] // block_rounds
    for block in range(block_count - 2):
        start, stop = (block + 2) * block_rounds, (block + 3) * block_rounds
        future[block * block_rounds : (block + 1) * block_rounds] = np.mean(
            class_probability[start:stop], axis=0
        )
    return future


def _resolved_config_hash(spec: PocSpec) -> str:
    profile_resolution: Mapping[str, object] = (
        {"dataset_profile": spec.dataset_profile, "pilot_partitions": spec.pilot_partitions}
        if spec.dataset_profile == "pilot"
        else {}
    )
    return canonical_digest(
        {
            "raw": spec.raw,
            "resolved": {
                "circuits": tuple(
                    {
                        "circuit_id": item.circuit_id,
                        "family": item.family,
                        "distance": item.distance,
                    }
                    for item in spec.circuits
                ),
                "condition_sets": spec.condition_sets,
                "component_bounds": spec.component_bounds,
                "dynamics": spec.dynamics,
                "public_root_seed": spec.public_root_seed,
                "trajectories_per_condition": spec.trajectories_per_condition,
                "burn_in_rounds": spec.burn_in_rounds,
                "scored_rounds": spec.scored_rounds,
                "episode_rounds": spec.episode_rounds,
                "block_rounds": spec.block_rounds,
                "target_config": spec.target_config,
                "fit_config": spec.fit_config,
                "forecast_samples": spec.forecast_samples,
                "forecast_reference_samples": spec.forecast_reference_samples,
                "forecast_mean_tolerance": spec.forecast_mean_tolerance,
                "bootstrap_resamples": spec.bootstrap_resamples,
                "confidence_level": spec.confidence_level,
                "retry_attempts": spec.retry_attempts,
                "chunk_rounds": spec.chunk_rounds,
                "roots": spec.roots,
                **profile_resolution,
            },
        }
    )


def assemble_artifacts(
    request: GenerationRequest,
    path: PhysicalNoisePath,
    sampled: SampledTrajectory,
    *,
    attempt: int,
) -> tuple[ObservableTrajectory, LabelTrajectory, Mapping[str, object]]:
    """Assemble auditable public and offline-only lanes from one deterministic sample."""
    spec, job = request.spec, request.job
    episode_map = tuple(
        build_memory_episode(
            job.circuit,
            path.component_probability[
                index * spec.episode_rounds : (index + 1) * spec.episode_rounds
            ],
            index,
            spec,
        )
        for index in range(path.component_probability.shape[0] // spec.episode_rounds)
    )
    truth = canonicalize_dem_truth(sampled.circuit, episode_map)
    global_round = np.arange(path.component_probability.shape[0], dtype=np.uint32)
    observable = ObservableTrajectory(
        detector_bits=sampled.detector_bits,
        detector_valid=sampled.detector_valid,
        logical_observable=sampled.logical_observable,
        global_round=global_round,
        episode=sampled.episode,
        round_in_episode=sampled.round_in_episode,
        block=global_round // np.uint32(spec.block_rounds),
        detector_role=sampled.detector_role,
        circuit_phase=sampled.circuit_phase,
        max_source_round=global_round.astype(np.int64),
    )
    labels = LabelTrajectory(
        component_probability=path.component_probability,
        latent_factor=path.latent_factor,
        class_probability=truth.class_probability,
        future_block_probability=_future_block_probability(
            truth.class_probability, spec.block_rounds
        ),
    )
    config_hash = _resolved_config_hash(spec)
    schema_value = spec.raw["schema_version"]
    if not isinstance(schema_value, int):
        raise InvalidPhysicalPath("invalid schema version")
    metadata_values: dict[str, object] = {
        "schema_version": schema_value,
        "condition_id": job.condition_id,
        "trajectory_id": job.trajectory_id,
        "split": job.split,
        "circuit_hash": hashlib.sha256(str(sampled.circuit).encode("utf-8")).hexdigest(),
        "undecomposed_dem_hash": truth.dem_hash,
        "canonical_catalog_hash": truth.catalog.catalog_hash,
        "canonical_catalog": {
            "class_count": len(truth.catalog.classes),
            "duplicate_sizes": truth.catalog.duplicate_sizes,
            "graphlike_mass": truth.catalog.graphlike_mass,
            "adaptable_mass": truth.catalog.adaptable_mass,
            "ambiguous_logical_mass": truth.catalog.ambiguous_logical_mass,
            "hyperedge_mass": truth.catalog.hyperedge_mass,
        },
        "dynamics_hash": canonical_digest(path.generator_metadata),
        "generation_law": {
            "dynamics_id": job.dynamics_id,
            "component_bounds": path.generator_metadata["component_bounds"],
            "missingness_parameters": dict(path.missingness_parameters),
            "observation_flip_probability": path.observation_flip_probability,
            "contamination_is_post_sampling": path.contamination_is_post_sampling,
        },
        "resolved_config_hash": config_hash,
        "package_versions": _package_versions(),
        "public_seed_commitment": canonical_digest({"public_root_seed": job.root_seed}),
        "stream_commitments": _stream_commitments(spec, job, attempt),
        "logical_array_shapes": {
            "detector_bits": tuple(observable.detector_bits.shape),
            "logical_observable": tuple(observable.logical_observable.shape),
            "class_probability": tuple(labels.class_probability.shape),
        },
        "episode_rounds": spec.episode_rounds,
        "block_rounds": spec.block_rounds,
        "creation_attempt": attempt,
    }
    if spec.dataset_profile == "pilot":
        metadata_values["dataset_profile"] = spec.dataset_profile
    metadata: Mapping[str, object] = MappingProxyType(metadata_values)
    return observable, labels, metadata


def generate_job(request: GenerationRequest) -> TrajectoryResult:
    """Run one transaction, converting only expected scientific/publication failures."""
    last_error: InvalidPhysicalPath | InvalidCircuit | ArtifactConflict | None = None
    for attempt in range(request.spec.retry_attempts):
        try:
            path = generate_dynamics(
                request.spec,
                request.job,
                request.spec.scored_rounds,
                request.spec.burn_in_rounds,
                attempt=attempt,
            )
            sampled = sample_trajectory(request.spec, request.job, path, attempt=attempt)
            observable, labels, metadata = assemble_artifacts(
                request, path, sampled, attempt=attempt
            )
            publish_trajectory(request.root, request.job, observable, labels, metadata)
            hashes = verify_trajectory_pair(
                request.root,
                request.job,
                resolved_config_hash=_resolved_config_hash(request.spec),
            )
            if hashes is None:
                raise ArtifactConflict("published artifact pair is missing")
            return TrajectoryResult.complete(request.job, *hashes)
        except (InvalidPhysicalPath, InvalidCircuit, ArtifactConflict) as error:
            last_error = error
    if isinstance(last_error, InvalidPhysicalPath):
        return TrajectoryResult.failed(request.job, FailureCode.INVALID_PATH, str(last_error))
    if isinstance(last_error, InvalidCircuit):
        return TrajectoryResult.failed(request.job, FailureCode.CIRCUIT_INVALID, str(last_error))
    return TrajectoryResult.failed(request.job, FailureCode.ARTIFACT_CONFLICT, str(last_error))


def _manifest_payload(
    results: Mapping[tuple[str, int], TrajectoryResult],
    spec: PocSpec,
    *,
    provenance: ManifestProvenance | None = None,
    sealed_commitment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    ordered = [results[key] for key in sorted(results)]
    payload: dict[str, object] = {
        "schema_version": 1,
        "resolved_config_hash": _resolved_config_hash(spec),
        "generation": {
            "trajectories_per_condition": spec.trajectories_per_condition,
            "burn_in_rounds": spec.burn_in_rounds,
            "scored_rounds": spec.scored_rounds,
        },
        "results": [
            {
                "condition_id": result.job_key[0],
                "trajectory_id": result.job_key[1],
                "completed": result.completed,
                "observable_hash": result.observable_hash,
                "label_hash": result.label_hash,
                "pair_id": result.pair_id,
                "failure": None
                if result.failure is None
                else {
                    "condition_id": result.failure.condition_id,
                    "trajectory_id": result.failure.trajectory_id,
                    "stage": result.failure.stage,
                    "code": result.failure.code.value,
                    "message": result.failure.message,
                },
            }
            for result in ordered
        ],
    }
    if spec.dataset_profile == "pilot":
        payload.update(
            {
                "dataset_profile": spec.dataset_profile,
                "expected_job_keys": _pilot_expected_job_keys(spec),
            }
        )
    if provenance is not None:
        payload["provenance"] = serialize_manifest_provenance(provenance)
    if sealed_commitment is not None:
        payload["sealed_commitment"] = dict(sealed_commitment)
    return payload


def _pilot_expected_job_keys(spec: PocSpec) -> list[list[str | int]]:
    return [
        [job.condition_id, job.trajectory_id]
        for job in sorted(
            expand_jobs(spec, include_sealed=True),
            key=lambda item: (item.condition_id, item.trajectory_id),
        )
    ]


def _run_manifest(
    results: Mapping[tuple[str, int], TrajectoryResult],
    manifest_hash: str,
    provenance: ManifestProvenance | None = None,
) -> RunManifest:
    ordered = [results[key] for key in sorted(results)]
    hashes = {
        f"{result.job_key[0]}:{result.job_key[1]}": (result.observable_hash, result.label_hash)
        for result in ordered
        if result.completed and result.observable_hash is not None and result.label_hash is not None
    }
    failures = tuple(result.failure for result in ordered if result.failure is not None)
    return RunManifest(
        generated=0,
        resumed=0,
        completed=len(hashes),
        trajectory_hashes=MappingProxyType(hashes),
        failures=failures,
        manifest_hash=manifest_hash,
        provenance=provenance,
    )


def _reject_private_seed_values(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if ("seed" in normalized and "commitment" not in normalized) or normalized == (
                "sealed_manifest"
            ):
                raise ArtifactConflict("run manifest contains private seed data")
            _reject_private_seed_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_seed_values(item)


def _manifest_sealed_commitment(
    spec: PocSpec, root: Path, provenance: ManifestProvenance | None
) -> Mapping[str, str] | None:
    if provenance is None:
        return None
    relative = spec.raw.get("sealed_commitment_path")
    if not isinstance(relative, str):
        raise ArtifactConflict("invalid sealed commitment path")
    path = root / relative
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactConflict("invalid sealed commitment") from error
    digest = value.get("digest") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"algorithm", "digest"}
        or value.get("algorithm") != "sha256"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactConflict("invalid sealed commitment")
    return MappingProxyType({"algorithm": "sha256", "digest": digest})


def select_incomplete_jobs(
    jobs: Sequence[TrajectoryJob],
    *,
    completed_job_keys: Collection[tuple[str, int]],
    job_limit: int | None,
) -> tuple[TrajectoryJob, ...]:
    """Select a stable bounded prefix without making scheduling order observable."""
    if job_limit is not None and job_limit < 1:
        raise ValueError("job limit must be positive")
    incomplete = tuple(
        job
        for job in sorted(jobs, key=lambda item: (item.condition_id, item.trajectory_id))
        if (job.condition_id, job.trajectory_id) not in completed_job_keys
    )
    return incomplete if job_limit is None else incomplete[:job_limit]


def _validate_execution_identity(
    spec: PocSpec,
    workers: int,
    execution_options: ExecutionOptions,
    provenance: ManifestProvenance | None,
) -> None:
    if execution_options.execution_backend == "local":
        if execution_options.job_limit is not None or execution_options.checkpoint_identity:
            raise ValueError("bounded execution options require the kaggle backend")
        if provenance is not None:
            raise ValueError("local generation does not accept kaggle provenance")
        return
    if spec.dataset_profile != "pilot":
        raise ValueError("kaggle generation requires the pilot dataset profile")
    if workers != 1:
        raise ValueError("kaggle generation requires exactly one worker")
    if execution_options.job_limit is None or execution_options.checkpoint_identity is None:
        raise ValueError("kaggle generation requires a job limit and checkpoint identity")
    if execution_options.generation_mode != "standard":
        raise ValueError("bounded scientific generation is not implemented")
    if (
        provenance is None
        or provenance.execution_backend != execution_options.execution_backend
        or provenance.checkpoint_identity != execution_options.checkpoint_identity
        or provenance.generation_mode != execution_options.generation_mode
        or provenance.generation_chunk_rounds != execution_options.generation_chunk_rounds
        or provenance.generation_law_version != STANDARD_GENERATION_LAW_VERSION
    ):
        raise ValueError("kaggle execution provenance mismatch")


def _previous_failures(
    manifest_path: Path, resolved_config_hash: str
) -> Mapping[tuple[str, int], TrajectoryResult]:
    if not manifest_path.exists():
        return MappingProxyType({})
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("results"), list):
        raise TypeError("invalid run manifest")
    if document.get("resolved_config_hash") != resolved_config_hash:
        return MappingProxyType({})
    failures: dict[tuple[str, int], TrajectoryResult] = {}
    for item in document["results"]:
        if (
            not isinstance(item, Mapping)
            or item.get("completed")
            or not isinstance(item.get("failure"), Mapping)
        ):
            continue
        failure = item["failure"]
        condition_id, trajectory_id = failure.get("condition_id"), failure.get("trajectory_id")
        code, message = failure.get("code"), failure.get("message")
        if (
            isinstance(condition_id, str)
            and isinstance(trajectory_id, int)
            and isinstance(code, str)
            and isinstance(message, str)
        ):
            failures[(condition_id, trajectory_id)] = TrajectoryResult.failed(
                TrajectoryJob(
                    condition_id,
                    trajectory_id,
                    "development",
                    CircuitSpec("repetition_d3", "repetition", 3),
                    "f01",
                    0,
                ),
                FailureCode(code),
                message,
            )
    return MappingProxyType(failures)


def _previous_completed(
    manifest_path: Path, resolved_config_hash: str
) -> Mapping[tuple[str, int], tuple[str, str, str] | None]:
    if not manifest_path.exists():
        return MappingProxyType({})
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("results"), list):
        raise TypeError("invalid run manifest")
    if document.get("resolved_config_hash") != resolved_config_hash:
        return MappingProxyType({})
    completed: dict[tuple[str, int], tuple[str, str, str] | None] = {}
    for item in document["results"]:
        if not isinstance(item, Mapping) or item.get("completed") is not True:
            continue
        condition_id, trajectory_id = item.get("condition_id"), item.get("trajectory_id")
        observable_hash, label_hash, pair_id = (
            item.get("observable_hash"),
            item.get("label_hash"),
            item.get("pair_id"),
        )
        if isinstance(condition_id, str) and isinstance(trajectory_id, int):
            completed[(condition_id, trajectory_id)] = (
                (observable_hash, label_hash, pair_id)
                if (
                    isinstance(observable_hash, str)
                    and isinstance(label_hash, str)
                    and isinstance(pair_id, str)
                )
                else None
            )
    return MappingProxyType(completed)


def assert_run_manifest_identity(
    root: Path, spec: PocSpec, provenance: ManifestProvenance | None = None
) -> None:
    """Reject a root already committed to another profile or resolved configuration."""
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        return
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactConflict("invalid existing run manifest") from error
    if not isinstance(document, Mapping):
        raise ArtifactConflict("invalid existing run manifest")
    _reject_private_seed_values(document)
    if document.get("dataset_profile", "production") != spec.dataset_profile or document.get(
        "resolved_config_hash"
    ) != _resolved_config_hash(spec):
        raise ArtifactConflict("run manifest profile or configuration mismatch")
    if spec.dataset_profile == "pilot":
        expected_generation = {
            "trajectories_per_condition": spec.trajectories_per_condition,
            "burn_in_rounds": spec.burn_in_rounds,
            "scored_rounds": spec.scored_rounds,
        }
        if document.get("generation") != expected_generation or document.get(
            "expected_job_keys"
        ) != _pilot_expected_job_keys(spec):
            raise ArtifactConflict("pilot manifest geometry or expected job keys mismatch")
    encoded_provenance = document.get("provenance")
    if provenance is None:
        if encoded_provenance is not None:
            raise ArtifactConflict("run manifest execution identity mismatch")
    else:
        try:
            existing_provenance = deserialize_manifest_provenance(encoded_provenance)
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactConflict("run manifest execution identity mismatch") from error
        if existing_provenance != provenance:
            raise ArtifactConflict("run manifest execution identity mismatch")


def generate_matrix(
    spec: PocSpec,
    jobs: Sequence[TrajectoryJob],
    root: Path,
    *,
    workers: int,
    execution_options: ExecutionOptions | None = None,
    provenance: ManifestProvenance | None = None,
) -> RunManifest:
    """Generate a sorted matrix with parent-owned atomic manifest replacement."""
    if workers < 1:
        raise ValueError("workers must be positive")
    options = execution_options or ExecutionOptions()
    _validate_execution_identity(spec, workers, options, provenance)
    assert_run_manifest_identity(root, spec, provenance)
    manifest_path = root / "run_manifest.json"
    config_hash = _resolved_config_hash(spec)
    sealed_commitment = _manifest_sealed_commitment(spec, root, provenance)
    previous = _previous_failures(manifest_path, config_hash)
    previous_completed = _previous_completed(manifest_path, config_hash)
    results: dict[tuple[str, int], TrajectoryResult] = {}
    request_jobs: list[TrajectoryJob] = []
    resumed = 0
    for job in sorted(jobs, key=lambda item: (item.condition_id, item.trajectory_id)):
        try:
            hashes = verify_trajectory_pair(root, job, resolved_config_hash=config_hash)
        except ArtifactConflict as error:
            results[(job.condition_id, job.trajectory_id)] = TrajectoryResult.failed(
                job, FailureCode.ARTIFACT_CONFLICT, str(error)
            )
            continue
        if hashes is not None:
            prior_hashes = previous_completed.get((job.condition_id, job.trajectory_id))
            if (
                job.condition_id,
                job.trajectory_id,
            ) in previous_completed and prior_hashes != hashes:
                results[(job.condition_id, job.trajectory_id)] = TrajectoryResult.failed(
                    job, FailureCode.ARTIFACT_CONFLICT, "manifest pair identity mismatch"
                )
                continue
            results[(job.condition_id, job.trajectory_id)] = TrajectoryResult.complete(job, *hashes)
            resumed += 1
        elif (job.condition_id, job.trajectory_id) in previous:
            results[(job.condition_id, job.trajectory_id)] = previous[
                (job.condition_id, job.trajectory_id)
            ]
        else:
            request_jobs.append(job)
    requests = [
        GenerationRequest(spec, job, root)
        for job in select_incomplete_jobs(
            request_jobs,
            completed_job_keys=(),
            job_limit=options.job_limit,
        )
    ]
    generated = 0
    if workers == 1:
        iterator = (generate_job(request) for request in requests)
        for result in iterator:
            results[result.job_key] = result
            if result.completed:
                generated += 1
            write_manifest(
                manifest_path,
                _manifest_payload(
                    results,
                    spec,
                    provenance=provenance,
                    sealed_commitment=sealed_commitment,
                ),
            )
    elif requests:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(generate_job, request) for request in requests]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results[result.job_key] = result
                if result.completed:
                    generated += 1
                write_manifest(
                    manifest_path,
                    _manifest_payload(
                        results,
                        spec,
                        provenance=provenance,
                        sealed_commitment=sealed_commitment,
                    ),
                )
    manifest_hash = write_manifest(
        manifest_path,
        _manifest_payload(
            results,
            spec,
            provenance=provenance,
            sealed_commitment=sealed_commitment,
        ),
    )
    base = _run_manifest(results, manifest_hash, provenance)
    return replace(base, generated=generated, resumed=resumed)


def generate_bounded_checkpoint(
    spec: PocSpec,
    jobs: Sequence[TrajectoryJob],
    root: Path,
    checkpoint_root: Path,
    *,
    workers: int,
    execution_options: ExecutionOptions,
    provenance: ManifestProvenance,
) -> RunManifest:
    """Generate a bounded verified prefix and export only after a new complete pair."""
    manifest = generate_matrix(
        spec,
        jobs,
        root,
        workers=workers,
        execution_options=execution_options,
        provenance=provenance,
    )
    if manifest.generated:
        export_checkpoint(
            root,
            checkpoint_root,
            expected_config_hash=_resolved_config_hash(spec),
            expected_provenance=provenance,
        )
    return manifest


def verify_dataset(
    spec: PocSpec, jobs: Sequence[TrajectoryJob], root: Path
) -> tuple[GateResult, ...]:
    """Verify existing public artifacts and report applicable gates without regeneration."""
    config_hash = _resolved_config_hash(spec)
    complete = True
    for job in jobs:
        try:
            complete = (
                verify_trajectory_pair(root, job, resolved_config_hash=config_hash) is not None
                and complete
            )
        except ArtifactConflict:
            complete = False
    context = AuditContext(
        reproducible_hashes=complete,
        circuit_dem_valid=complete,
        stationary_rate_difference=0.0,
        stationary_rate_tolerance=0.01,
        physical_probability_valid=complete,
        episode_indices_isolated=complete,
        duplicate_composition_exact=complete,
        ambiguity_and_hyperedge_reported=complete,
        target_identifiable=None,
        observation_budget_difference=0.0,
        observation_budget_tolerance=0.01,
        codrift_expected_sign=1,
        codrift_observed_covariance=0.1,
        pre_onset_auc=0.5,
        pre_onset_monte_carlo_half_width=0.01,
        loaders_and_splits_isolated=complete,
    )
    return run_dataset_gates(context)
