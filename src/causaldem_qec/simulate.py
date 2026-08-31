from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np
import stim  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from causaldem_qec.core import CircuitSpec, PocSpec, TrajectoryJob, derive_seed


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
