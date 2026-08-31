from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType

import numpy as np
import stim  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from causaldem_qec.core import (
    CanonicalCatalog,
    CanonicalClass,
    CanonicalDemTruth,
    CircuitSpec,
    PocSpec,
    TrajectoryJob,
    derive_seed,
)


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
        raise ValueError("invalid physical probability bounds")
    probability = np.asarray(lower + (upper - lower) * expit(latent), dtype=np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("nonfinite physical path")
    if np.any(probability <= lower) or np.any(probability >= upper):
        raise ValueError("saturated bounded transform")
    return probability


def xor_compose(probabilities: np.ndarray) -> float:
    """Return the probability that an odd number of independent faults occurs."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values >= 0.5)):
        raise ValueError("invalid duplicate probabilities")
    return float(-0.5 * np.expm1(np.log1p(-2.0 * values).sum()))


def _dem_errors(dem: stim.DetectorErrorModel) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
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
        decoder_compatible = len(logical_signatures) == 1 and all(member[2] for member in members)
        classes.append(
            CanonicalClass(
                class_id=class_id,
                detector_signature=signature,
                logical_signatures=logical_signatures,
                probability=probability,
                support_size=len(signature),
                graphlike=graphlike,
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
            continue
        locations = explanation_sources.get((detector_ids, observable_ids), [])
        source_instruction = locations.pop(0) if locations else None
        source_episode = None
        source_round = None
        if source_instruction is not None:
            for candidate_episode, (start, end, episode) in enumerate(spans):
                if start <= source_instruction < end:
                    source_episode = candidate_episode
                    for round_index, (round_start, round_end) in enumerate(episode.round_instruction_ranges):
                        if round_start <= source_instruction - start < round_end:
                            source_round = round_index
                            break
                    break
        detector_episodes = {detector_map[item][0] for item in detector_ids}
        observable_episodes = set(observable_ids)
        if len(detector_episodes) != 1 or (observable_episodes and observable_episodes != detector_episodes):
            raise InvalidCircuit("DEM event crosses an episode boundary")
        episode_index = next(iter(detector_episodes))
        if source_episode != episode_index or source_round is None:
            signature = tuple(
                sorted((detector_map[item][1], detector_map[item][2], detector_map[item][3]) for item in detector_ids)
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


def canonicalize_dem(circuit: stim.Circuit, episode_map: Sequence[BuiltEpisode]) -> CanonicalCatalog:
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
            by_round_class.setdefault((source_round, class_index[signature]), []).append(event_probability)
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
        ("DQ04", context.physical_probability_valid, {"physical_probability_valid": context.physical_probability_valid}),
        ("DQ05", context.episode_indices_isolated, {"episode_indices_isolated": context.episode_indices_isolated}),
        ("DQ06", context.duplicate_composition_exact, {"duplicate_composition_exact": context.duplicate_composition_exact}),
        ("DQ07", context.ambiguity_and_hyperedge_reported, {"ambiguity_and_hyperedge_reported": context.ambiguity_and_hyperedge_reported}),
        ("DQ12", context.loaders_and_splits_isolated, {"loaders_and_splits_isolated": context.loaders_and_splits_isolated}),
    )
    results = [
        GateResult(gate_id, GateStatus.PASS if passed else GateStatus.FAIL, MappingProxyType(evidence), ())
        for gate_id, passed, evidence in simple
    ]
    results.extend(
        (
            GateResult("DQ03", GateStatus.PASS if context.stationary_rate_difference <= context.stationary_rate_tolerance else GateStatus.FAIL, MappingProxyType({"difference": context.stationary_rate_difference, "tolerance": context.stationary_rate_tolerance}), ()),
            GateResult("DQ08", GateStatus.NOT_RUN, MappingProxyType({"target_identifiable": "not_run"}), ()),
            GateResult("DQ09", GateStatus.PASS if context.observation_budget_difference <= context.observation_budget_tolerance else GateStatus.FAIL, MappingProxyType({"difference": context.observation_budget_difference, "tolerance": context.observation_budget_tolerance}), ()),
            GateResult("DQ10", GateStatus.PASS if context.codrift_expected_sign * context.codrift_observed_covariance > 0.0 else GateStatus.FAIL, MappingProxyType({"expected_sign": context.codrift_expected_sign, "observed_covariance": context.codrift_observed_covariance}), ()),
            GateResult("DQ11", GateStatus.PASS if abs(context.pre_onset_auc - 0.5) <= context.pre_onset_monte_carlo_half_width else GateStatus.FAIL, MappingProxyType({"auc": context.pre_onset_auc, "monte_carlo_half_width": context.pre_onset_monte_carlo_half_width}), ()),
        )
    )
    by_id = {item.gate_id: item for item in results}
    return tuple(by_id[gate_id] for gate_id, _ in DATASET_GATES)


def dataset_gates_complete(results: Sequence[GateResult]) -> bool:
    """A verification gate is complete only when every registered audit passed."""
    return len(results) == len(DATASET_GATES) and all(item.status is GateStatus.PASS for item in results)


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
    task = "repetition_code:memory" if circuit.family == "repetition" else "surface_code:rotated_memory_z"
    return stim.Circuit.generated(task, distance=circuit.distance, rounds=32).flattened()


def _target_values(targets: list[stim.GateTarget]) -> tuple[int, ...]:
    return tuple(target.qubit_value for target in targets)


def _component_operations(circuit: CircuitSpec, instruction: stim.CircuitInstruction) -> tuple[str, ...]:
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


def _layout_bounds(
    spec: PocSpec | None, bound_kind: str
) -> tuple[float, float]:
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


def _rng(spec: PocSpec, job: TrajectoryJob, stream: str) -> np.random.Generator:
    schema_version = _integer(spec.raw["schema_version"], "schema_version")
    seed = derive_seed(
        job.root_seed,
        schema_version,
        job.condition_id,
        job.trajectory_id,
        stream,
        0,
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
    spec: PocSpec, job: TrajectoryJob, total_rounds: int, components: tuple[PhysicalComponent, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = spec.dynamics["f03"]
    shared_phi = _config_pair(spec, "f03", "shared_phi")
    dynamics_rng = _rng(spec, job, "dynamics")
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
        [_number(type_sign[NOISE_KIND[component.kind][1]], "f03.type_sign") for component in components],
        dtype=np.float64,
    )
    geometry = np.linspace(-1.0, 1.0, len(components), dtype=np.float64)
    if len(components) > 1:
        geometry /= np.linalg.norm(geometry)
    loadings = np.column_stack(
        [
            np.full(len(components), _number(config["global_loading"], "f03.global_loading")),
            _number(config["x_loading"], "f03.x_loading") * geometry,
        ]
    ) * signs[:, None]
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


def _canonical_component_pairs(components: tuple[PhysicalComponent, ...]) -> tuple[tuple[int, int], ...]:
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
) -> PhysicalNoisePath:
    scored = spec.scored_rounds if scored_rounds is None else scored_rounds
    discarded = spec.burn_in_rounds if burn_in is None else burn_in
    if scored <= 0 or discarded < 0:
        raise InvalidPhysicalPath("round counts must be positive after burn-in")
    components = component_layout(job.circuit, spec)
    lower = np.asarray([component.lower for component in components], dtype=np.float64)
    upper = np.asarray([component.upper for component in components], dtype=np.float64)
    total = scored + discarded
    dynamics_rng = _rng(spec, job, "dynamics")
    match job.dynamics_id:
        case "f01":
            offsets = dynamics_rng.normal(scale=_config_float(spec, "f01", "offset_sd"), size=len(components))
            all_latent = np.broadcast_to(offsets, (total, len(components))).copy()
            all_factors = all_latent[:, : min(2, len(components))]
        case "f02":
            all_latent = stationary_ar1(
                _config_float(spec, "f02", "phi"), (total, len(components)), dynamics_rng
            ) * _config_float(spec, "f02", "loading")
            all_factors = all_latent[:, : min(2, len(components))]
        case "f03" | "f07" | "f08" | "f12":
            all_latent, all_factors, loadings = _f03_latent(spec, job, total, components)
            if job.dynamics_id == "f12":
                config = spec.dynamics["f12"]
                burst = _geometric_run_mask(
                    total,
                    _number(config["onset_hazard"], "f12.onset_hazard"),
                    _number(config["mean_duration"], "f12.mean_duration"),
                    _rng(spec, job, "burst"),
                )
                all_latent += _number(config["amplitude"], "f12.amplitude") * burst[:, None] * loadings.sum(axis=1)
        case "f06":
            periods = np.linspace(
                _config_float(spec, "f06", "start_period"),
                _config_float(spec, "f06", "stop_period"),
                total,
            )
            phase = dynamics_rng.uniform(0.0, 2.0 * np.pi)
            chirp = np.sin(2.0 * np.pi * np.cumsum(1.0 / periods) + phase)
            all_latent = _config_float(spec, "f06", "amplitude") * chirp[:, None]
            all_factors = np.column_stack((chirp, np.cos(2.0 * np.pi * np.cumsum(1.0 / periods) + phase)))
        case "f14_positive" | "f14_negative":
            config = spec.dynamics[job.dynamics_id]
            pairs = _canonical_component_pairs(components)
            pair_count = len(pairs)
            driver = stationary_ar1(_number(config["phi"], "f14.phi"), (total, pair_count), dynamics_rng)
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


def append_noise(circuit: stim.Circuit, name: str, targets: list[stim.GateTarget], p: float) -> None:
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
    circuit: CircuitSpec, round_probability: np.ndarray, episode_id: int
) -> BuiltEpisode:
    components = component_layout(circuit)
    if round_probability.dtype != np.dtype(np.float64) or round_probability.shape != (32, len(components)):
        raise InvalidCircuit("round probabilities must be float64 with one row per round and component")
    if not np.isfinite(round_probability).all() or np.any((round_probability <= 0.0) | (round_probability >= 0.5)):
        raise InvalidCircuit("round probabilities must be finite physical probabilities")
    lookup = {(component.kind, component.targets): component.component_id for component in components}
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
            if circuit.family == "surface" and instruction.name in {"MR", "M"} and kind == "measure_z_basis":
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
        bits.shape[0], parameters["burst_hazard"], parameters["mean_duration"], rng)
    subset = np.asarray(rng.random(bits.shape[1]) < parameters["detector_fraction"], dtype=np.bool_)
    if not subset.any():
        subset[rng.integers(bits.shape[1])] = True
    missing |= burst[:, None] & subset[None, :]
    valid[present & missing] = False
    return valid


def sample_trajectory(spec: PocSpec, job: TrajectoryJob, path: PhysicalNoisePath) -> SampledTrajectory:
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
        0,
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
        detector_bits, present, path.missingness_parameters, _rng(spec, job, "missingness")
    )
    if path.contamination_is_post_sampling:
        flips = _rng(spec, job, "contamination").random(detector_bits.shape) < path.observation_flip_probability
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
