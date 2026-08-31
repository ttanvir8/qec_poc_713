from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from causaldem_qec.core import CircuitSpec, TrajectoryJob, load_spec
from causaldem_qec.simulate import (
    NOISE_KIND,
    bounded_probability,
    component_layout,
    generate_dynamics,
)


@pytest.fixture
def job_for_family():
    circuits = {
        "repetition_d3": CircuitSpec("repetition_d3", "repetition", 3),
        "surface_d3": CircuitSpec("surface_d3", "surface", 3),
    }

    def make(family: str, circuit_id: str = "repetition_d3") -> TrajectoryJob:
        sealed = {"f12", "f14_positive", "f14_negative"}
        return TrajectoryJob(
            condition_id=f"{circuit_id}__{family}",
            trajectory_id=0,
            split="sealed_test" if family in sealed else "development",
            circuit=circuits[circuit_id],
            dynamics_id=family,
            root_seed=713,
        )

    return make


@pytest.mark.parametrize(
    "family",
    ["f01", "f02", "f03", "f06", "f07", "f08", "f12", "f14_positive", "f14_negative"],
)
def test_dynamic_family_is_deterministic_finite_and_bounded(family, job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    first = generate_dynamics(spec, job_for_family(family), scored_rounds=1024, burn_in=128)
    second = generate_dynamics(spec, job_for_family(family), scored_rounds=1024, burn_in=128)
    np.testing.assert_array_equal(first.component_probability, second.component_probability)
    assert np.isfinite(first.component_probability).all()
    assert (first.component_probability > first.lower_bound).all()
    assert (first.component_probability < first.upper_bound).all()


def test_codrift_modes_have_opposite_covariance_signs(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    positive = generate_dynamics(spec, job_for_family("f14_positive"), 4096, 256)
    negative = generate_dynamics(spec, job_for_family("f14_negative"), 4096, 256)
    assert np.cov(positive.latent_factor[:, 0], positive.latent_factor[:, 1], bias=True)[0, 1] > 0
    assert np.cov(negative.latent_factor[:, 0], negative.latent_factor[:, 1], bias=True)[0, 1] < 0


def test_observation_corruption_does_not_change_physical_truth(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    contaminated = generate_dynamics(spec, job_for_family("f08"), 1024, 128)
    assert contaminated.observation_flip_probability == pytest.approx(0.01)
    assert contaminated.contamination_is_post_sampling


from causaldem_qec.simulate import build_memory_episode, sample_trajectory


@pytest.mark.parametrize("circuit_id", ["repetition_d3", "repetition_d5", "surface_d3", "surface_d5"])
def test_expanded_episode_compiles_without_repeat(circuit_id) -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == circuit_id)
    rates = np.full((32, len(component_layout(circuit))), 0.001, dtype=np.float64)
    built = build_memory_episode(circuit, rates, episode_id=7)
    assert "REPEAT" not in str(built.circuit)
    dem = built.circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=False,
        flatten_loops=True,
    )
    assert dem.num_detectors == built.circuit.num_detectors
    assert built.circuit.num_observables == 8


def test_detector_sampler_returns_observable_from_same_shot(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    job = job_for_family("f03", "surface_d3")
    path = generate_dynamics(spec, job, scored_rounds=96, burn_in=32)
    sampled = sample_trajectory(spec, job, path)
    assert sampled.detector_bits.shape[0] == 96
    assert sampled.logical_observable.shape == (3,)
    assert sampled.episode.tolist() == np.repeat(np.arange(3), 32).tolist()
    assert sampled.round_in_episode.tolist() == np.tile(np.arange(32), 3).tolist()


def test_surface_cx_injects_one_correlated_error_per_pair() -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "surface_d3")
    rates = np.full((32, len(component_layout(circuit))), 0.001, dtype=np.float64)
    built = build_memory_episode(circuit, rates, episode_id=0)
    template_pairs = sum(
        len(instruction.targets_copy()) // 2
        for instruction in built.circuit
        if instruction.name == "CX"
    )
    correlated = [instruction for instruction in built.circuit if instruction.name == "E"]
    assert len(correlated) == template_pairs
    assert all(len(instruction.targets_copy()) == 2 for instruction in correlated)


def test_repetition_data_noise_never_targets_measurement_ancillas() -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "repetition_d3")
    rates = np.full((32, len(component_layout(circuit))), 0.001, dtype=np.float64)
    built = build_memory_episode(circuit, rates, episode_id=0)
    data_qubits = {
        target.qubit_value
        for instruction in built.circuit
        if instruction.name == "M"
        for target in instruction.targets_copy()
    }
    for instruction in built.circuit:
        if instruction.name == "DEPOLARIZE1":
            assert {target.qubit_value for target in instruction.targets_copy()} <= data_qubits


def test_surface_tick_has_data_idle_noise_location() -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "surface_d3")
    rates = np.full((32, len(component_layout(circuit))), 0.001, dtype=np.float64)
    built = build_memory_episode(circuit, rates, episode_id=0)
    instructions = list(built.circuit)
    assert any(
        instruction.name == "TICK" and following.name == "DEPOLARIZE1"
        for instruction, following in pairwise(instructions)
    )


def test_f14_layout_pairs_each_physical_type_by_canonical_motif() -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "surface_d3")
    counts = Counter(NOISE_KIND[component.kind][1] for component in component_layout(circuit))
    assert all(count % 2 == 0 for count in counts.values())


def test_generate_dynamics_reads_component_bounds_from_spec(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    bounds = dict(spec.component_bounds)
    bounds["repetition_data"] = (0.001, 0.002)
    path = generate_dynamics(replace(spec, component_bounds=bounds), job_for_family("f01"), 64, 8)
    assert np.all(path.lower_bound[[0, 1]] == 0.001)
    assert np.all(path.upper_bound[[0, 1]] == 0.002)


def test_bounded_probability_rejects_saturated_transform() -> None:
    with pytest.raises(ValueError, match="saturated"):
        bounded_probability(
            np.asarray([[-np.inf]], dtype=np.float64),
            np.asarray([0.001], dtype=np.float64),
            np.asarray([0.002], dtype=np.float64),
        )


def test_built_episode_preserves_detector_coordinate_phase_and_built_ranges() -> None:
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "surface_d3")
    rates = np.full((32, len(component_layout(circuit))), 0.001, dtype=np.float64)
    built = build_memory_episode(circuit, rates, episode_id=0)
    assert built.detector_phase.max() == 32
    assert built.round_instruction_ranges[-1][1] == len(list(built.circuit))


def test_sampled_circuit_phase_is_the_episode_round(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    job = job_for_family("f03", "surface_d3")
    sampled = sample_trajectory(spec, job, generate_dynamics(spec, job, 64, 8))
    np.testing.assert_array_equal(sampled.circuit_phase, np.tile(np.arange(32, dtype=np.uint8), 2))


def test_generator_metadata_contains_resolved_reproduction_parameters(job_for_family) -> None:
    spec = load_spec(Path("configs/poc.json"))
    path = generate_dynamics(spec, job_for_family("f07"), 64, 8)
    assert path.generator_metadata["resolved_parameters"] == spec.dynamics["f07"]
    assert path.generator_metadata["base_parameters"] == spec.dynamics["f03"]
