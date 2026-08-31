from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import stim

from causaldem_qec.core import CircuitSpec, TrajectoryJob, expand_jobs, load_spec
from causaldem_qec.simulate import (
    NOISE_KIND,
    AuditContext,
    GateStatus,
    bounded_probability,
    canonicalize_dem,
    canonicalize_dem_truth,
    canonicalize_test_dem,
    component_layout,
    dataset_gates_complete,
    generate_dynamics,
    generate_matrix,
    run_dataset_gates,
    xor_compose,
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


@pytest.mark.parametrize(
    "circuit_id", ["repetition_d3", "repetition_d5", "surface_d3", "surface_d5"]
)
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


def test_duplicate_detector_signature_uses_exact_odd_parity() -> None:
    actual = xor_compose(np.array([0.10, 0.20], dtype=np.float64))
    assert actual == pytest.approx(0.10 * 0.80 + 0.90 * 0.20)
    assert actual != pytest.approx(0.30)


def test_equal_detectors_with_different_logicals_are_static_only() -> None:
    dem = stim.DetectorErrorModel("error(0.01) D0\nerror(0.02) D0 L0")
    catalog = canonicalize_test_dem(dem, detector_round={0: 0})
    assert len(catalog.classes) == 1
    assert not catalog.classes[0].decoder_compatible
    assert catalog.ambiguous_logical_mass > 0.0


def test_hyperedge_is_measured_but_not_adaptable() -> None:
    dem = stim.DetectorErrorModel("error(0.03) D0 D1 D2")
    catalog = canonicalize_test_dem(dem, detector_round={0: 0, 1: 0, 2: 1})
    assert catalog.hyperedge_mass == pytest.approx(0.03)
    assert not catalog.classes[0].adaptable


@pytest.fixture
def auditable_tiny_trajectory() -> AuditContext:
    return AuditContext(
        reproducible_hashes=True,
        circuit_dem_valid=True,
        stationary_rate_difference=0.001,
        stationary_rate_tolerance=0.01,
        physical_probability_valid=True,
        episode_indices_isolated=True,
        duplicate_composition_exact=True,
        ambiguity_and_hyperedge_reported=True,
        target_identifiable=None,
        observation_budget_difference=0.001,
        observation_budget_tolerance=0.01,
        codrift_expected_sign=1,
        codrift_observed_covariance=0.2,
        pre_onset_auc=0.5,
        pre_onset_monte_carlo_half_width=0.03,
        loaders_and_splits_isolated=True,
    )


def test_all_twelve_quality_gates_are_named(auditable_tiny_trajectory: AuditContext) -> None:
    results = run_dataset_gates(auditable_tiny_trajectory)
    assert [result.gate_id for result in results] == [f"DQ{i:02d}" for i in range(1, 13)]
    assert results[7].status is GateStatus.NOT_RUN
    assert all(
        result.status is GateStatus.PASS for index, result in enumerate(results) if index != 7
    )
    assert not dataset_gates_complete(results)


def test_wrong_codrift_sign_invalidates_condition(auditable_tiny_trajectory: AuditContext) -> None:
    broken = replace(
        auditable_tiny_trajectory,
        codrift_expected_sign=-1,
        codrift_observed_covariance=0.2,
    )
    result = run_dataset_gates(broken)[9]
    assert result.gate_id == "DQ10"
    assert result.status is GateStatus.FAIL


def test_pre_onset_feature_signal_fails_exogenous_burst_gate(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    broken = replace(
        auditable_tiny_trajectory,
        pre_onset_auc=0.75,
        pre_onset_monte_carlo_half_width=0.03,
    )
    result = run_dataset_gates(broken)[10]
    assert result.gate_id == "DQ11"
    assert result.status is GateStatus.FAIL


def test_complete_trajectory_dem_truth_is_round_resolved_and_frozen(job_for_family) -> None:
    circuit_spec = job_for_family("f03").circuit
    rates = np.full((32, len(component_layout(circuit_spec))), 0.001, dtype=np.float64)
    episodes = tuple(
        build_memory_episode(circuit_spec, rates, episode_id) for episode_id in range(2)
    )
    truth = canonicalize_dem_truth(
        sum((episode.circuit for episode in episodes), stim.Circuit()), episodes
    )
    assert truth.class_probability.shape == (64, len(truth.catalog.classes))
    assert np.any(truth.class_probability == 0.0)
    assert truth.dem_hash
    assert truth.catalog.catalog_hash


def test_detectorless_dem_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="no detector support"):
        canonicalize_test_dem(stim.DetectorErrorModel("error(0.01) L0"), detector_round={})
    circuit = stim.Circuit("X_ERROR(0.01) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    with pytest.raises(ValueError, match="no detector support"):
        canonicalize_dem(circuit, ())


def test_computed_audit_inputs_produce_gate_evidence(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    circuit = stim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    present = np.ones((32, 2), dtype=np.bool_)
    valid = present.copy()
    valid[:2, 0] = False
    context = replace(
        auditable_tiny_trajectory,
        stationary_circuit=circuit,
        stationary_shots=256,
        audit_seed=713,
        observation_present=present,
        observation_valid=valid,
        observation_flip_mask=np.zeros_like(present),
        observation_expected_mcar=2.0 / 64.0,
        codrift_samples=np.column_stack((np.arange(32), np.arange(32))).astype(np.float64),
        codrift_marginal_tolerance=0.01,
        pre_onset_scores=np.arange(32, dtype=np.float64),
        pre_onset_labels=np.tile(np.array([False, True]), 16),
    )
    results = {item.gate_id: item for item in run_dataset_gates(context)}
    assert "detector_rate" in results["DQ03"].evidence
    assert "mcar_rate" in results["DQ09"].evidence
    assert "marginal_range_difference" in results["DQ10"].evidence
    assert "permutation_threshold" in results["DQ11"].evidence


def test_observation_gate_checks_mcar_burst_and_flip_budgets(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    present = np.ones((10, 2), dtype=np.bool_)
    mcar = np.zeros_like(present)
    mcar[0, 0] = True
    burst = np.zeros_like(present)
    burst[1:4, 1] = True
    valid = present & ~(mcar | burst)
    flips = np.zeros_like(present)
    flips[:5, 0] = True
    result = run_dataset_gates(
        replace(
            auditable_tiny_trajectory,
            observation_present=present,
            observation_valid=valid,
            observation_mcar_mask=mcar,
            observation_burst_mask=burst,
            observation_flip_mask=flips,
            observation_expected_mcar=0.05,
            observation_expected_burst=0.0,
            observation_expected_flip=0.0,
            observation_rate_tolerance=0.01,
        )
    )[8]
    assert result.status is GateStatus.FAIL
    assert result.evidence["burst_rate"] == pytest.approx(0.15)
    assert result.evidence["flip_rate"] == pytest.approx(0.25)


def test_observation_gate_separates_independent_mcar_from_combined_burst_validity(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    present = np.ones((5, 2), dtype=np.bool_)
    mcar = np.zeros_like(present)
    mcar[0, 0] = True
    burst = np.zeros_like(present)
    burst[1:3, 1] = True
    valid = present & ~(mcar | burst)
    result = run_dataset_gates(
        replace(
            auditable_tiny_trajectory,
            observation_present=present,
            observation_valid=valid,
            observation_mcar_mask=mcar,
            observation_burst_mask=burst,
            observation_expected_mcar=0.1,
            observation_expected_burst=0.2,
            observation_rate_tolerance=0.001,
        )
    )[8]
    assert result.status is GateStatus.PASS
    assert result.evidence["mcar_rate"] == pytest.approx(0.1)
    assert result.evidence["burst_rate"] == pytest.approx(0.2)


def test_observation_gate_fails_the_specific_independent_mcar_budget(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    present = np.ones((5, 2), dtype=np.bool_)
    mcar = np.zeros_like(present)
    mcar[:2, 0] = True
    burst = np.zeros_like(present)
    burst[1:3, 1] = True
    valid = present & ~(mcar | burst)
    result = run_dataset_gates(
        replace(
            auditable_tiny_trajectory,
            observation_present=present,
            observation_valid=valid,
            observation_mcar_mask=mcar,
            observation_burst_mask=burst,
            observation_expected_mcar=0.1,
            observation_expected_burst=0.2,
            observation_rate_tolerance=0.001,
        )
    )[8]
    assert result.status is GateStatus.FAIL
    assert result.evidence["mcar_rate"] == pytest.approx(0.2)
    assert result.evidence["burst_rate"] == pytest.approx(0.2)


def test_codrift_uses_its_own_marginal_tolerance(auditable_tiny_trajectory: AuditContext) -> None:
    samples = np.column_stack((np.arange(20), np.arange(20) * 2.0)).astype(np.float64)
    result = run_dataset_gates(
        replace(auditable_tiny_trajectory, codrift_samples=samples, codrift_marginal_tolerance=0.1)
    )[9]
    assert result.status is GateStatus.FAIL


def test_codrift_samples_require_a_dedicated_marginal_tolerance(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    samples = np.column_stack((np.arange(20), np.arange(20))).astype(np.float64)
    with pytest.raises(ValueError, match="codrift_marginal_tolerance"):
        run_dataset_gates(replace(auditable_tiny_trajectory, codrift_samples=samples))


def test_pre_onset_gate_uses_maximum_feature_departure_deterministically(
    auditable_tiny_trajectory: AuditContext,
) -> None:
    labels = np.tile(np.array([False, True]), 16)
    features = np.column_stack((np.zeros(32), labels.astype(np.float64)))
    context = replace(
        auditable_tiny_trajectory,
        pre_onset_feature_scores=features,
        pre_onset_labels=labels,
        audit_seed=713,
    )
    first = run_dataset_gates(context)[10]
    second = run_dataset_gates(context)[10]
    assert first.status is GateStatus.FAIL
    assert first.evidence["max_abs_departure"] == second.evidence["max_abs_departure"]
    assert first.evidence["permutation_threshold"] == second.evidence["permutation_threshold"]


@pytest.fixture
def tiny_spec():
    spec = load_spec(Path("configs/poc.json"))
    circuit = next(item for item in spec.circuits if item.circuit_id == "repetition_d3")
    return replace(
        spec,
        circuits=(circuit,),
        condition_sets={"distance_3": ("f01",), "distance_5": ()},
        trajectories_per_condition=2,
        burn_in_rounds=32,
        scored_rounds=256,
    )


@pytest.mark.slow
def test_worker_count_does_not_change_artifact_hashes(tiny_spec, tmp_path: Path) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)
    one = generate_matrix(tiny_spec, jobs, tmp_path / "one", workers=1)
    two = generate_matrix(tiny_spec, reversed(jobs), tmp_path / "two", workers=2)
    assert one.trajectory_hashes == two.trajectory_hashes


def test_resume_skips_only_verified_complete_trajectory(tiny_spec, tmp_path: Path) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)
    first = generate_matrix(tiny_spec, jobs, tmp_path, workers=1)
    second = generate_matrix(tiny_spec, jobs, tmp_path, workers=1)
    assert second.generated == 0
    assert second.resumed == first.completed


def test_failed_trajectory_remains_in_manifest(tiny_spec, tmp_path: Path) -> None:
    bad_bounds = dict(tiny_spec.component_bounds)
    bad_bounds["repetition_data"] = (0.6, 0.7)
    invalid = replace(tiny_spec, component_bounds=bad_bounds)
    jobs = expand_jobs(invalid, include_sealed=False)[:1]
    manifest = generate_matrix(invalid, jobs, tmp_path, workers=1)
    assert manifest.completed == 0
    assert manifest.failures[0].trajectory_id == 0
