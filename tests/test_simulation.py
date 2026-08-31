from pathlib import Path

import numpy as np
import pytest

from causaldem_qec.core import CircuitSpec, TrajectoryJob, load_spec
from causaldem_qec.simulate import component_layout, generate_dynamics


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
