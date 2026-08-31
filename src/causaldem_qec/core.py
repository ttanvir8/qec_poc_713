from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

Split = Literal["train", "validation", "id_test", "development", "sealed_test"]

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "public_root_seed",
        "sealed_commitment_path",
        "rounds",
        "trajectories_per_condition",
        "circuits",
        "condition_sets",
        "component_bounds",
        "dynamics",
        "target",
        "fit",
        "forecast",
        "evaluation",
        "runtime",
        "roots",
    }
)
_CIRCUIT_IDS = frozenset({"repetition_d3", "repetition_d5", "surface_d3", "surface_d5"})
_DYNAMICS_IDS = frozenset(
    {"f01", "f02", "f03", "f06", "f07", "f08", "f12", "f14_positive", "f14_negative"}
)
_COMPONENT_IDS = frozenset(
    {
        "repetition_data",
        "repetition_measure",
        "surface_1q",
        "surface_2q",
        "surface_reset",
        "surface_measure",
        "surface_correlated",
    }
)


class FailureCode(StrEnum):
    INVALID_CONFIG = "invalid_config"
    INVALID_PATH = "invalid_path"
    ARTIFACT_CONFLICT = "artifact_conflict"
    CIRCUIT_INVALID = "circuit_invalid"
    QUALITY_GATE = "quality_gate"
    CAUSAL_VIOLATION = "causal_violation"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True, slots=True)
class CircuitSpec:
    circuit_id: str
    family: Literal["repetition", "surface"]
    distance: Literal[3, 5]


@dataclass(frozen=True, slots=True)
class TrajectoryJob:
    condition_id: str
    trajectory_id: int
    split: Split
    circuit: CircuitSpec
    dynamics_id: str
    root_seed: int


@dataclass(frozen=True, slots=True)
class TrajectoryFailure:
    condition_id: str
    trajectory_id: int
    stage: str
    code: FailureCode
    message: str


@dataclass(frozen=True, slots=True)
class PocSpec:
    raw: Mapping[str, object]
    circuits: tuple[CircuitSpec, ...]
    condition_sets: Mapping[str, tuple[str, ...]]
    component_bounds: Mapping[str, tuple[float, float]]
    dynamics: Mapping[str, Mapping[str, object]]
    public_root_seed: int
    trajectories_per_condition: int
    burn_in_rounds: int
    scored_rounds: int
    episode_rounds: int
    block_rounds: int
    target_config: Mapping[str, float | int]
    fit_config: Mapping[str, tuple[float, ...]]
    forecast_samples: int
    forecast_reference_samples: int
    forecast_mean_tolerance: float
    bootstrap_resamples: int
    confidence_level: float
    retry_attempts: int
    chunk_rounds: int
    roots: Mapping[str, str]


def _stable_words(parts: tuple[str | int, ...]) -> list[int]:
    digest = sha256("\x1f".join(map(str, parts)).encode()).digest()
    return list(np.frombuffer(digest, dtype="<u4"))


def derive_seed(root_seed: int, *parts: str | int) -> np.random.SeedSequence:
    return np.random.SeedSequence([root_seed, *_stable_words(parts)])


def source_cutoff(block: int, block_rounds: int) -> int:
    if block < 0 or block_rounds <= 0:
        raise ValueError("block must be nonnegative and block_rounds positive")
    return (block + 1) * block_rounds - 1


def target_interval(block: int, block_rounds: int) -> tuple[int, int]:
    if block < 0 or block_rounds <= 0:
        raise ValueError("block must be nonnegative and block_rounds positive")
    first = (block + 2) * block_rounds
    return first, first + block_rounds


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if extra:
        raise ValueError(f"unknown {name} keys: {', '.join(extra)}")
    if missing:
        raise ValueError(f"missing {name} keys: {', '.join(missing)}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object, name: str, *, lower: float | None = None, upper: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    number = float(value)
    if (
        not isfinite(number)
        or (lower is not None and number <= lower)
        or (upper is not None and number >= upper)
    ):
        raise ValueError(f"{name} is outside its valid range")
    return number


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")  # noqa: TRY004
    return value


def _probability(value: object, name: str) -> float:
    probability = _number(value, name, upper=1)
    if probability < 0:
        raise ValueError(f"{name} is outside its valid range")
    return probability


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze(value))


def _numeric_values(value: object, name: str) -> tuple[float, ...]:
    items = _list(value, name)
    if not items:
        raise ValueError(f"{name} must not be empty")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))


def _validate_dynamics(dynamics: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    _exact_keys(dynamics, _DYNAMICS_IDS, "dynamics")
    expected_keys = {
        "f01": frozenset({"offset_sd"}),
        "f02": frozenset({"phi", "loading"}),
        "f03": frozenset(
            {"shared_phi", "local_phi", "local_sd", "global_loading", "x_loading", "type_sign"}
        ),
        "f06": frozenset({"amplitude", "start_period", "stop_period"}),
        "f07": frozenset({"base", "mcar", "burst_hazard", "mean_duration", "detector_fraction"}),
        "f08": frozenset({"base", "flip_probability"}),
        "f12": frozenset({"base", "onset_hazard", "mean_duration", "amplitude"}),
        "f14_positive": frozenset({"phi", "amplitude", "sign"}),
        "f14_negative": frozenset({"phi", "amplitude", "sign"}),
    }
    validated: dict[str, Mapping[str, object]] = {}
    for dynamics_id, keys in expected_keys.items():
        entry = _mapping(dynamics[dynamics_id], f"dynamics.{dynamics_id}")
        _exact_keys(entry, keys, f"dynamics.{dynamics_id}")
        validated[dynamics_id] = _freeze_mapping(entry)

    f01 = _mapping(dynamics["f01"], "dynamics.f01")
    _number(f01["offset_sd"], "f01.offset_sd", lower=0)
    f02 = _mapping(dynamics["f02"], "dynamics.f02")
    _number(f02["phi"], "f02.phi", lower=-1, upper=1)
    _number(f02["loading"], "f02.loading", lower=0)
    f03 = _mapping(dynamics["f03"], "dynamics.f03")
    shared_phi = _numeric_values(f03["shared_phi"], "f03.shared_phi")
    if len(shared_phi) != 2 or any(abs(phi) >= 1 for phi in shared_phi):
        raise ValueError("f03.shared_phi must contain two stationary coefficients")
    _number(f03["local_phi"], "f03.local_phi", lower=-1, upper=1)
    for key in ("local_sd", "global_loading", "x_loading"):
        _number(f03[key], f"f03.{key}", lower=0)
    type_sign = _mapping(f03["type_sign"], "f03.type_sign")
    _exact_keys(type_sign, _COMPONENT_IDS, "f03.type_sign")
    if any(
        _integer(value, f"f03.type_sign.{key}", minimum=-1) not in {-1, 1}
        for key, value in type_sign.items()
    ):
        raise ValueError("f03.type_sign values must be -1 or 1")
    f06 = _mapping(dynamics["f06"], "dynamics.f06")
    _number(f06["amplitude"], "f06.amplitude", lower=0)
    start_period = _integer(f06["start_period"], "f06.start_period")
    stop_period = _integer(f06["stop_period"], "f06.stop_period")
    if start_period <= stop_period:
        raise ValueError("f06.start_period must exceed f06.stop_period")
    f07 = _mapping(dynamics["f07"], "dynamics.f07")
    if _string(f07["base"], "f07.base") != "f03":
        raise ValueError("f07.base must be f03")
    for key in ("mcar", "burst_hazard", "detector_fraction"):
        _probability(f07[key], f"f07.{key}")
    _integer(f07["mean_duration"], "f07.mean_duration")
    f08 = _mapping(dynamics["f08"], "dynamics.f08")
    if _string(f08["base"], "f08.base") != "f03":
        raise ValueError("f08.base must be f03")
    _probability(f08["flip_probability"], "f08.flip_probability")
    f12 = _mapping(dynamics["f12"], "dynamics.f12")
    if _string(f12["base"], "f12.base") != "f03":
        raise ValueError("f12.base must be f03")
    _probability(f12["onset_hazard"], "f12.onset_hazard")
    _integer(f12["mean_duration"], "f12.mean_duration")
    _number(f12["amplitude"], "f12.amplitude", lower=0)
    for dynamics_id, expected_sign in (("f14_positive", 1), ("f14_negative", -1)):
        entry = _mapping(dynamics[dynamics_id], f"dynamics.{dynamics_id}")
        _number(entry["phi"], f"{dynamics_id}.phi", lower=-1, upper=1)
        _number(entry["amplitude"], f"{dynamics_id}.amplitude", lower=0)
        if _integer(entry["sign"], f"{dynamics_id}.sign", minimum=-1) != expected_sign:
            raise ValueError(f"{dynamics_id}.sign must be {expected_sign}")
    return MappingProxyType(validated)


def load_spec(path: Path) -> PocSpec:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read config: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config: {path}") from error
    config = _mapping(loaded, "config")
    _exact_keys(config, _TOP_LEVEL_KEYS, "config")

    if _integer(config["schema_version"], "schema_version") != 1:
        raise ValueError("schema_version must be 1")
    public_root_seed = _integer(config["public_root_seed"], "public_root_seed", minimum=0)
    _string(config["sealed_commitment_path"], "sealed_commitment_path")
    trajectories_per_condition = _integer(
        config["trajectories_per_condition"], "trajectories_per_condition"
    )
    if trajectories_per_condition != 64:
        raise ValueError("trajectories_per_condition must be 64")

    rounds = _mapping(config["rounds"], "rounds")
    _exact_keys(rounds, frozenset({"burn_in", "scored", "episode", "block"}), "rounds")
    burn_in_rounds = _integer(rounds["burn_in"], "rounds.burn_in")
    scored_rounds = _integer(rounds["scored"], "rounds.scored")
    episode_rounds = _integer(rounds["episode"], "rounds.episode")
    block_rounds = _integer(rounds["block"], "rounds.block")
    if (burn_in_rounds, scored_rounds, episode_rounds, block_rounds) != (4096, 65536, 32, 256):
        raise ValueError("rounds must match the committed trajectory fidelity")

    circuits_config = _mapping(config["circuits"], "circuits")
    _exact_keys(circuits_config, _CIRCUIT_IDS, "circuits")
    circuits: list[CircuitSpec] = []
    for circuit_id in sorted(circuits_config):
        entry = _mapping(circuits_config[circuit_id], f"circuits.{circuit_id}")
        _exact_keys(entry, frozenset({"family", "distance"}), f"circuits.{circuit_id}")
        family_value = _string(entry["family"], f"circuits.{circuit_id}.family")
        if family_value not in {"repetition", "surface"}:
            raise ValueError(f"circuits.{circuit_id}.family is invalid")
        distance_value = _integer(entry["distance"], f"circuits.{circuit_id}.distance")
        if distance_value not in {3, 5}:
            raise ValueError(f"circuits.{circuit_id}.distance is invalid")
        expected_id = f"{family_value}_d{distance_value}"
        if circuit_id != expected_id:
            raise ValueError(f"circuit id must match family and distance: {circuit_id}")
        circuits.append(
            CircuitSpec(
                circuit_id,
                cast(Literal["repetition", "surface"], family_value),
                cast(Literal[3, 5], distance_value),
            )
        )

    condition_sets_config = _mapping(config["condition_sets"], "condition_sets")
    _exact_keys(condition_sets_config, frozenset({"distance_3", "distance_5"}), "condition_sets")
    condition_sets: dict[str, tuple[str, ...]] = {}
    expected_conditions = {
        "distance_3": (
            "f01",
            "f02",
            "f03",
            "f06",
            "f07",
            "f08",
            "f12",
            "f14_positive",
            "f14_negative",
        ),
        "distance_5": ("f01", "f03", "f06", "f12", "f14_positive", "f14_negative"),
    }
    for key, expected in expected_conditions.items():
        condition_values = tuple(
            _string(value, f"condition_sets.{key}")
            for value in _list(condition_sets_config[key], key)
        )
        if condition_values != expected:
            raise ValueError(f"condition_sets.{key} is not the committed condition matrix")
        condition_sets[key] = condition_values

    bounds_config = _mapping(config["component_bounds"], "component_bounds")
    _exact_keys(bounds_config, _COMPONENT_IDS, "component_bounds")
    component_bounds: dict[str, tuple[float, float]] = {}
    for component_id in sorted(bounds_config):
        bound_values = _numeric_values(
            bounds_config[component_id], f"component_bounds.{component_id}"
        )
        if len(bound_values) != 2 or not 0 < bound_values[0] < bound_values[1] < 0.5:
            raise ValueError(f"component_bounds.{component_id} must be two values in (0, 0.5)")
        component_bounds[component_id] = (bound_values[0], bound_values[1])

    dynamics = _validate_dynamics(_mapping(config["dynamics"], "dynamics"))

    target = _mapping(config["target"], "target")
    _exact_keys(
        target,
        frozenset(
            {
                "max_classes",
                "max_queries",
                "max_query_weight",
                "max_lag",
                "relative_singular_tolerance",
                "moment_floor",
            }
        ),
        "target",
    )
    target_config: Mapping[str, float | int] = MappingProxyType(
        {
            "max_classes": _integer(target["max_classes"], "target.max_classes"),
            "max_queries": _integer(target["max_queries"], "target.max_queries"),
            "max_query_weight": _integer(target["max_query_weight"], "target.max_query_weight"),
            "max_lag": _integer(target["max_lag"], "target.max_lag", minimum=0),
            "relative_singular_tolerance": _number(
                target["relative_singular_tolerance"], "target.relative_singular_tolerance", lower=0
            ),
            "moment_floor": _number(
                target["moment_floor"], "target.moment_floor", lower=0, upper=1
            ),
        }
    )

    fit = _mapping(config["fit"], "fit")
    _exact_keys(fit, frozenset({"ewma", "ar_phi", "innovation_sd", "damping"}), "fit")
    fit_config = MappingProxyType(
        {key: _numeric_values(fit[key], f"fit.{key}") for key in sorted(fit)}
    )
    if (
        any(value <= 0 for value in fit_config["ewma"])
        or any(abs(value) >= 1 for value in fit_config["ar_phi"])
        or any(value <= 0 for value in fit_config["innovation_sd"])
        or any(value <= 0 for value in fit_config["damping"])
    ):
        raise ValueError("fit values are outside their valid ranges")

    forecast = _mapping(config["forecast"], "forecast")
    _exact_keys(forecast, frozenset({"samples", "reference_samples", "mean_tolerance"}), "forecast")
    forecast_samples = _integer(forecast["samples"], "forecast.samples")
    forecast_reference_samples = _integer(
        forecast["reference_samples"], "forecast.reference_samples"
    )
    forecast_mean_tolerance = _number(
        forecast["mean_tolerance"], "forecast.mean_tolerance", lower=0
    )

    evaluation = _mapping(config["evaluation"], "evaluation")
    _exact_keys(evaluation, frozenset({"bootstrap_resamples", "interval"}), "evaluation")
    bootstrap_resamples = _integer(
        evaluation["bootstrap_resamples"], "evaluation.bootstrap_resamples"
    )
    confidence_level = _number(evaluation["interval"], "evaluation.interval", lower=0, upper=1)

    runtime = _mapping(config["runtime"], "runtime")
    _exact_keys(runtime, frozenset({"retry_attempts", "chunk_rounds"}), "runtime")
    retry_attempts = _integer(runtime["retry_attempts"], "runtime.retry_attempts")
    if retry_attempts > 3:
        raise ValueError("retry_attempts must not exceed 3")
    chunk_rounds = _integer(runtime["chunk_rounds"], "runtime.chunk_rounds")

    roots_config = _mapping(config["roots"], "roots")
    _exact_keys(roots_config, frozenset({"data", "runs", "reports"}), "roots")
    roots = MappingProxyType(
        {key: _string(roots_config[key], f"roots.{key}") for key in sorted(roots_config)}
    )

    return PocSpec(
        raw=_freeze_mapping(config),
        circuits=tuple(circuits),
        condition_sets=MappingProxyType(condition_sets),
        component_bounds=MappingProxyType(component_bounds),
        dynamics=dynamics,
        public_root_seed=public_root_seed,
        trajectories_per_condition=trajectories_per_condition,
        burn_in_rounds=burn_in_rounds,
        scored_rounds=scored_rounds,
        episode_rounds=episode_rounds,
        block_rounds=block_rounds,
        target_config=target_config,
        fit_config=fit_config,
        forecast_samples=forecast_samples,
        forecast_reference_samples=forecast_reference_samples,
        forecast_mean_tolerance=forecast_mean_tolerance,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        retry_attempts=retry_attempts,
        chunk_rounds=chunk_rounds,
        roots=roots,
    )


def _split_for(dynamics_id: str, trajectory_id: int) -> Split:
    if dynamics_id in {"f01", "f02", "f03"}:
        if trajectory_id < 32:
            return "train"
        if trajectory_id < 48:
            return "validation"
        return "id_test"
    if dynamics_id in {"f06", "f07", "f08"}:
        return "development"
    return "sealed_test"


def expand_jobs(spec: PocSpec, *, include_sealed: bool) -> tuple[TrajectoryJob, ...]:
    jobs: list[TrajectoryJob] = []
    for circuit in spec.circuits:
        dynamics_ids = spec.condition_sets[f"distance_{circuit.distance}"]
        for dynamics_id in sorted(dynamics_ids):
            split = _split_for(dynamics_id, 0)
            if split == "sealed_test" and not include_sealed:
                continue
            condition_id = f"{circuit.circuit_id}__{dynamics_id}"
            for trajectory_id in range(spec.trajectories_per_condition):
                jobs.append(
                    TrajectoryJob(
                        condition_id=condition_id,
                        trajectory_id=trajectory_id,
                        split=_split_for(dynamics_id, trajectory_id),
                        circuit=circuit,
                        dynamics_id=dynamics_id,
                        root_seed=spec.public_root_seed,
                    )
                )
    return tuple(jobs)
