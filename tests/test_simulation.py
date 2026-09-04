import json
import shutil
from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import stim

from causaldem_qec.artifacts import load_sealed_seed, write_sealed_commitment
from causaldem_qec.cli import main
from causaldem_qec.core import (
    CircuitSpec,
    ExecutionOptions,
    ManifestProvenance,
    TrajectoryJob,
    expand_jobs,
    load_spec,
)
from causaldem_qec.simulate import (
    BOUNDED_GENERATION_LAW_VERSION,
    BOUNDED_SAMPLING_LAW_VERSION,
    NOISE_KIND,
    AuditContext,
    GateStatus,
    GenerationRequest,
    _future_block_probability,
    _manifest_payload,
    assemble_artifacts,
    assert_run_manifest_identity,
    bounded_probability,
    canonicalize_bounded_dem_truth,
    canonicalize_dem,
    canonicalize_dem_truth,
    canonicalize_test_dem,
    component_layout,
    dataset_gates_complete,
    episode_fault_seed,
    generate_bounded_checkpoint,
    generate_dynamics,
    generate_dynamics_bounded,
    generate_matrix,
    process_memory_snapshot,
    run_dataset_gates,
    select_incomplete_jobs,
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


from causaldem_qec.simulate import (
    build_memory_episode,
    sample_trajectory,
    sample_trajectory_bounded,
)


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


def test_bounded_sampling_is_invariant_to_chunk_grouping() -> None:
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    path = generate_dynamics(spec, job, scored_rounds=256, burn_in=32)

    small_chunks = sample_trajectory_bounded(spec, job, path, chunk_rounds=32, attempt=1)
    large_chunks = sample_trajectory_bounded(spec, job, path, chunk_rounds=64, attempt=1)

    np.testing.assert_array_equal(small_chunks.detector_bits, large_chunks.detector_bits)
    np.testing.assert_array_equal(small_chunks.detector_valid, large_chunks.detector_valid)
    np.testing.assert_array_equal(small_chunks.logical_observable, large_chunks.logical_observable)


def test_bounded_sampling_releases_the_trajectory_circuit() -> None:
    """Catch bounded sampling retaining the monolithic Stim circuit after episode sampling."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    path = generate_dynamics(spec, job, scored_rounds=64, burn_in=8)

    sampled = sample_trajectory_bounded(spec, job, path, chunk_rounds=32, attempt=1)

    assert sampled.circuit is None


def test_bounded_sampling_writes_observations_to_memory_maps(tmp_path: Path) -> None:
    """Catch bounded sampling retaining detector lanes as trajectory-sized heap arrays."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    path = generate_dynamics(spec, job, scored_rounds=64, burn_in=8)

    sampled = sample_trajectory_bounded(spec, job, path, chunk_rounds=32, staging_root=tmp_path)

    assert isinstance(sampled.detector_bits, np.memmap)
    assert isinstance(sampled.detector_valid, np.memmap)
    assert (tmp_path / "detector_bits.npy").is_file()
    assert (tmp_path / "detector_valid.npy").is_file()


def test_episode_fault_seed_is_stable_and_unique_per_episode() -> None:
    """Catch a bounded run accidentally reusing a trajectory-wide fault seed."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")

    first = episode_fault_seed(spec, job, attempt=1, episode_index=3)
    repeated = episode_fault_seed(spec, job, attempt=1, episode_index=3)
    next_episode = episode_fault_seed(spec, job, attempt=1, episode_index=4)

    assert first == repeated
    assert first != next_episode


def test_bounded_metadata_commits_the_new_generation_and_sampling_laws() -> None:
    """Catch bounded artifacts that can be confused with the historical dataset law."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    request = GenerationRequest(
        spec,
        job,
        Path("."),
        generation_mode="bounded",
        generation_chunk_rounds=32,
    )
    path = generate_dynamics_bounded(spec, job, chunk_rounds=32)
    sampled = sample_trajectory_bounded(spec, job, path, chunk_rounds=32)

    _, _, metadata = assemble_artifacts(request, path, sampled, attempt=0)

    assert metadata["generation_law"]["generation_law_version"] == BOUNDED_GENERATION_LAW_VERSION
    assert metadata["generation_law"]["sampling_law_version"] == BOUNDED_SAMPLING_LAW_VERSION


def test_bounded_catalog_aggregates_all_episode_dem_events() -> None:
    """Catch a bounded catalog derived from only the first episode's event probabilities."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    path = generate_dynamics_bounded(spec, job, chunk_rounds=32)
    request = GenerationRequest(spec, job, Path("."), generation_mode="bounded", generation_chunk_rounds=32)
    sampled = sample_trajectory_bounded(spec, job, path, chunk_rounds=32)

    _, _, metadata = assemble_artifacts(request, path, sampled, attempt=0)
    first_episode = build_memory_episode(job.circuit, path.component_probability[:32], 0, spec)
    first_catalog = canonicalize_dem(first_episode.circuit, (first_episode,))

    assert metadata["canonical_catalog_hash"] != first_catalog.catalog_hash


def test_bounded_catalog_writes_class_probabilities_to_a_memory_map(tmp_path: Path) -> None:
    """Catch the bounded DEM second pass allocating its complete class lane on the heap."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    path = generate_dynamics_bounded(spec, job, chunk_rounds=32)

    truth = canonicalize_bounded_dem_truth(spec, job, path, staging_root=tmp_path)

    assert isinstance(truth.class_probability, np.memmap)
    assert (tmp_path / "class_probability.npy").is_file()


def test_future_block_probability_writes_to_a_memory_map(tmp_path: Path) -> None:
    """Catch forecast labels allocating a second trajectory-sized heap array."""
    source = np.arange(64, dtype=np.float64).reshape(32, 2)

    future = _future_block_probability(source, 8, staging_path=tmp_path / "future.npy")

    assert isinstance(future, np.memmap)
    assert (tmp_path / "future.npy").is_file()


def test_process_memory_snapshot_reports_positive_peak_rss() -> None:
    """Catch bounded-job diagnostics omitting the RSS value needed for Kaggle safety checks."""
    snapshot = process_memory_snapshot()

    assert snapshot["peak_rss_bytes"] > 0


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


def test_incomplete_jobs_are_selected_in_stable_sorted_order(tiny_spec) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)
    selected = select_incomplete_jobs(
        tuple(reversed(jobs)),
        completed_job_keys={(jobs[1].condition_id, jobs[1].trajectory_id)},
        job_limit=1,
    )

    assert selected == (jobs[0],)


def _tiny_pilot_spec():
    spec = load_spec(Path("configs/poc_pilot.json"))
    return replace(spec, burn_in_rounds=32, scored_rounds=256)


def _kaggle_execution() -> tuple[ExecutionOptions, ManifestProvenance]:
    options = ExecutionOptions(
        execution_backend="kaggle",
        job_limit=1,
        checkpoint_identity="owner/causaldem-pilot-checkpoint",
        checkpoint_version="owner/causaldem-pilot-checkpoint@7",
    )
    provenance = ManifestProvenance(
        source_commit="task-3-test",
        execution_backend="kaggle",
        generation_law_version="standard_monolithic_v1",
        checkpoint_identity="owner/causaldem-pilot-checkpoint",
    )
    return options, provenance


def test_bounded_generation_is_deterministic_across_clean_and_resumed_runs(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    jobs = expand_jobs(spec, include_sealed=False)[:2]
    options, provenance = _kaggle_execution()

    first = generate_matrix(
        spec,
        tuple(reversed(jobs)),
        tmp_path / "resumed",
        workers=1,
        execution_options=options,
        provenance=provenance,
    )
    second = generate_matrix(
        spec,
        tuple(reversed(jobs)),
        tmp_path / "resumed",
        workers=1,
        execution_options=options,
        provenance=provenance,
    )
    clean = generate_matrix(
        spec,
        tuple(reversed(jobs)),
        tmp_path / "clean",
        workers=1,
        execution_options=replace(options, job_limit=2),
        provenance=provenance,
    )

    assert first.generated == 1
    assert second.generated == 1
    assert second.resumed == 1
    assert second.trajectory_hashes == clean.trajectory_hashes


def test_bounded_and_standard_generation_have_distinct_f01_artifact_hashes(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    bounded_options = ExecutionOptions(
        generation_mode="bounded",
        generation_chunk_rounds=64,
    )

    standard = generate_matrix(spec, (job,), tmp_path / "standard", workers=1)
    bounded = generate_matrix(
        spec,
        (job,),
        tmp_path / "bounded",
        workers=1,
        execution_options=bounded_options,
    )

    assert bounded.trajectory_hashes != standard.trajectory_hashes
    manifest = json.loads((tmp_path / "bounded" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["mode"] == "bounded"
    assert manifest["generation"]["chunk_rounds"] == 64
    assert manifest["generation"]["generation_law_version"] == BOUNDED_GENERATION_LAW_VERSION
    assert manifest["generation"]["sampling_law_version"] == BOUNDED_SAMPLING_LAW_VERSION


def test_f01_bounded_dynamics_streams_chunks_without_standard_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")
    expected = generate_dynamics(spec, job, attempt=0)

    def fail_standard(*args: object, **kwargs: object) -> object:
        raise AssertionError("bounded f01 generation called the standard fallback")

    monkeypatch.setattr("causaldem_qec.simulate.generate_dynamics", fail_standard)
    actual = generate_dynamics_bounded(spec, job, chunk_rounds=64, attempt=0)

    np.testing.assert_array_equal(actual.component_probability, expected.component_probability)
    np.testing.assert_array_equal(actual.latent_factor, expected.latent_factor)
    assert actual.generator_metadata == expected.generator_metadata


def test_bounded_dynamics_writes_final_lanes_as_memory_maps(tmp_path: Path) -> None:
    """Catch bounded dynamics collecting all chunk arrays before returning its final lanes."""
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f01")

    path = generate_dynamics_bounded(spec, job, chunk_rounds=32, staging_root=tmp_path)

    assert isinstance(path.component_probability, np.memmap)
    assert isinstance(path.latent_factor, np.memmap)
    assert (tmp_path / "component_probability.npy").is_file()
    assert (tmp_path / "latent_factor.npy").is_file()


@pytest.mark.parametrize("dynamics_id", ["f02", "f14_positive", "f14_negative"])
def test_ar_bounded_dynamics_matches_standard_stream(
    dynamics_id: str,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(
        job for job in expand_jobs(spec, include_sealed=True) if job.dynamics_id == dynamics_id
    )

    expected = generate_dynamics(spec, job, attempt=1)
    actual = generate_dynamics_bounded(spec, job, chunk_rounds=64, attempt=1)

    np.testing.assert_array_equal(actual.component_probability, expected.component_probability)
    np.testing.assert_array_equal(actual.latent_factor, expected.latent_factor)
    assert actual.generator_metadata == expected.generator_metadata


def test_f03_bounded_factor_dynamics_matches_standard_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == "f03")

    expected = generate_dynamics(spec, job, attempt=1)
    monkeypatch.setattr(
        "causaldem_qec.simulate.generate_dynamics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bounded f03 generation called the standard fallback")
        ),
    )
    actual = generate_dynamics_bounded(spec, job, chunk_rounds=64, attempt=1)

    np.testing.assert_array_equal(actual.component_probability, expected.component_probability)
    np.testing.assert_array_equal(actual.latent_factor, expected.latent_factor)
    assert actual.generator_metadata == expected.generator_metadata


@pytest.mark.parametrize("dynamics_id", ["f07", "f08"])
def test_derived_factor_bounded_dynamics_matches_standard_stream(
    dynamics_id: str,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(
        job for job in expand_jobs(spec, include_sealed=False) if job.dynamics_id == dynamics_id
    )

    expected = generate_dynamics(spec, job, attempt=1)
    actual = generate_dynamics_bounded(spec, job, chunk_rounds=64, attempt=1)

    np.testing.assert_array_equal(actual.component_probability, expected.component_probability)
    np.testing.assert_array_equal(actual.latent_factor, expected.latent_factor)
    assert actual.generator_metadata == expected.generator_metadata
    assert actual.missingness_parameters == expected.missingness_parameters
    assert actual.observation_flip_probability == expected.observation_flip_probability


@pytest.mark.parametrize("dynamics_id", ["f06", "f12"])
def test_remaining_bounded_dynamics_match_standard_without_fallback(
    dynamics_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_pilot_spec()
    job = next(
        job for job in expand_jobs(spec, include_sealed=True) if job.dynamics_id == dynamics_id
    )

    expected = generate_dynamics(spec, job, attempt=1)
    monkeypatch.setattr(
        "causaldem_qec.simulate.generate_dynamics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"bounded {dynamics_id} generation called the standard fallback")
        ),
    )
    actual = generate_dynamics_bounded(spec, job, chunk_rounds=64, attempt=1)

    np.testing.assert_array_equal(actual.component_probability, expected.component_probability)
    np.testing.assert_array_equal(actual.latent_factor, expected.latent_factor)
    assert actual.generator_metadata == expected.generator_metadata


def test_nonsealed_bounded_resume_preserves_verified_sealed_manifest_entry(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    all_jobs = expand_jobs(spec, include_sealed=True)
    sealed_job = next(job for job in all_jobs if job.split == "sealed_test")
    nonsealed_job = next(job for job in all_jobs if job.split != "sealed_test")
    commitment = tmp_path / "data" / "manifests" / "sealed_commitment.json"
    commitment.parent.mkdir(parents=True)
    commitment.write_text(json.dumps({"algorithm": "sha256", "digest": "c" * 64}), encoding="utf-8")

    sealed = generate_matrix(
        spec,
        (sealed_job,),
        tmp_path,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )
    commitment.unlink()
    resumed = generate_matrix(
        spec,
        (nonsealed_job,),
        tmp_path,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )

    document = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    completed_keys = {
        (item["condition_id"], item["trajectory_id"])
        for item in document["results"]
        if item["completed"] is True
    }
    assert sealed.completed == 1
    assert resumed.completed == 2
    assert document["sealed_commitment"] == {
        "algorithm": "sha256",
        "digest": "c" * 64,
    }
    assert completed_keys == {
        (sealed_job.condition_id, sealed_job.trajectory_id),
        (nonsealed_job.condition_id, nonsealed_job.trajectory_id),
    }


def test_bounded_resume_rejects_a_new_disagreeing_sealed_commitment(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    all_jobs = expand_jobs(spec, include_sealed=True)
    sealed_job = next(job for job in all_jobs if job.split == "sealed_test")
    nonsealed_job = next(job for job in all_jobs if job.split != "sealed_test")
    commitment = tmp_path / "data" / "manifests" / "sealed_commitment.json"
    commitment.parent.mkdir(parents=True)
    commitment.write_text(json.dumps({"algorithm": "sha256", "digest": "c" * 64}), encoding="utf-8")
    generate_matrix(
        spec,
        (sealed_job,),
        tmp_path,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )
    manifest_path = tmp_path / "run_manifest.json"
    original_manifest = manifest_path.read_bytes()
    commitment.write_text(json.dumps({"algorithm": "sha256", "digest": "d" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed commitment mismatch"):
        generate_matrix(
            spec,
            (nonsealed_job,),
            tmp_path,
            workers=1,
            execution_options=options,
            provenance=provenance,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert not (
        tmp_path
        / "data"
        / "observable"
        / nonsealed_job.split
        / nonsealed_job.condition_id
        / str(nonsealed_job.trajectory_id)
    ).exists()


def test_kaggle_manifest_binds_execution_identity_without_raw_seed(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    generate_matrix(
        spec,
        expand_jobs(spec, include_sealed=False)[:1],
        tmp_path,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"] == {
        "source_commit": "task-3-test",
        "execution_backend": "kaggle",
        "generation_law_version": "standard_monolithic_v1",
        "checkpoint_identity": "owner/causaldem-pilot-checkpoint",
        "generation_mode": "standard",
        "generation_chunk_rounds": None,
    }
    assert manifest["checkpoint_input_version"] == "owner/causaldem-pilot-checkpoint@7"
    assert "root_seed" not in json.dumps(manifest)


def test_bounded_generation_exports_only_after_a_verified_pair(tmp_path: Path) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    output_root = tmp_path / "run"
    checkpoint_root = tmp_path / "checkpoint"

    manifest = generate_bounded_checkpoint(
        spec,
        expand_jobs(spec, include_sealed=False)[:1],
        output_root,
        checkpoint_root,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )

    assert manifest.generated == 1
    assert (checkpoint_root / "run_manifest.json").read_bytes() == (
        output_root / "run_manifest.json"
    ).read_bytes()
    assert len(tuple(checkpoint_root.glob("data/observable/*/*/*"))) == 1
    assert len(tuple(checkpoint_root.glob("data/labels/*/*/*"))) == 1


def test_sealed_checkpoint_export_resumes_with_public_commitment_only(tmp_path: Path) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    private = tmp_path / "private" / "sealed.json"
    private.parent.mkdir()
    private.write_text('{"root_seed":99887766}', encoding="utf-8")
    first_run = tmp_path / "first-run"
    commitment = first_run / "data" / "manifests" / "sealed_commitment.json"
    write_sealed_commitment(private, commitment)
    all_jobs = expand_jobs(spec, include_sealed=True)
    sealed_job = replace(
        next(job for job in all_jobs if job.split == "sealed_test"),
        root_seed=99887766,
    )
    nonsealed_job = next(job for job in all_jobs if job.split != "sealed_test")
    first_export = tmp_path / "checkpoint-1"

    generate_bounded_checkpoint(
        spec,
        (sealed_job,),
        first_run,
        first_export,
        workers=1,
        execution_options=options,
        provenance=provenance,
    )

    resumed_run = tmp_path / "resumed-run"
    shutil.copytree(first_export, resumed_run)
    resumed_commitment = resumed_run / "data" / "manifests" / "sealed_commitment.json"
    assert load_sealed_seed(private, resumed_commitment, purpose="sealed_evaluation") == 99887766
    second_export = tmp_path / "checkpoint-2"
    next_version_options = replace(
        options,
        checkpoint_version="owner/causaldem-pilot-checkpoint@8",
    )
    resumed = generate_bounded_checkpoint(
        spec,
        (sealed_job, nonsealed_job),
        resumed_run,
        second_export,
        workers=1,
        execution_options=next_version_options,
        provenance=provenance,
    )

    assert resumed.completed == 2
    resumed_manifest = json.loads((resumed_run / "run_manifest.json").read_text(encoding="utf-8"))
    assert resumed_manifest["provenance"]["checkpoint_identity"] == (
        "owner/causaldem-pilot-checkpoint"
    )
    assert resumed_manifest["checkpoint_input_version"] == ("owner/causaldem-pilot-checkpoint@8")
    assert (second_export / "data" / "manifests" / "sealed_commitment.json").is_file()
    assert not any(path.name == private.name for path in second_export.rglob("*"))


def test_kaggle_resume_rejects_private_seed_manifest_before_generation(tmp_path: Path) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    manifest = _manifest_payload(
        {},
        spec,
        provenance=provenance,
        checkpoint_input_version=options.checkpoint_version,
    )
    manifest["sealed_manifest"] = {"root_seed": 99887766}
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="private seed"):
        generate_matrix(
            spec,
            expand_jobs(spec, include_sealed=False)[:1],
            tmp_path,
            workers=1,
            execution_options=options,
            provenance=provenance,
        )
    assert not (tmp_path / "data").exists()


def test_kaggle_generation_rejects_invalid_sealed_commitment_before_worker(
    tmp_path: Path,
) -> None:
    spec = _tiny_pilot_spec()
    options, provenance = _kaggle_execution()
    commitment = tmp_path / "data" / "manifests" / "sealed_commitment.json"
    commitment.parent.mkdir(parents=True)
    commitment.write_text(json.dumps({"algorithm": "sha256", "digest": "z" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid sealed commitment"):
        generate_matrix(
            spec,
            expand_jobs(spec, include_sealed=False)[:1],
            tmp_path,
            workers=1,
            execution_options=options,
            provenance=provenance,
        )
    assert not (tmp_path / "data" / "observable").exists()


def test_failed_trajectory_remains_in_manifest(tiny_spec, tmp_path: Path) -> None:
    bad_bounds = dict(tiny_spec.component_bounds)
    bad_bounds["repetition_data"] = (0.6, 0.7)
    invalid = replace(tiny_spec, component_bounds=bad_bounds)
    jobs = expand_jobs(invalid, include_sealed=False)[:1]
    manifest = generate_matrix(invalid, jobs, tmp_path, workers=1)
    assert manifest.completed == 0
    assert manifest.failures[0].trajectory_id == 0


def test_manifest_binds_common_pair_id_to_completed_lane_hashes(tiny_spec, tmp_path: Path) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)[:1]
    generate_matrix(tiny_spec, jobs, tmp_path, workers=1)
    result = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))["results"][0]
    observable = json.loads(
        (
            tmp_path
            / "data"
            / "observable"
            / jobs[0].split
            / jobs[0].condition_id
            / "0"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    labels = json.loads(
        (
            tmp_path
            / "data"
            / "labels"
            / jobs[0].split
            / jobs[0].condition_id
            / "0"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert result["pair_id"] == observable["pair_id"] == labels["pair_id"]
    summary = labels["metadata"]["canonical_catalog"]
    assert summary["class_count"] > 0
    assert len(summary["duplicate_sizes"]) == summary["class_count"]
    assert set(summary) >= {
        "graphlike_mass",
        "adaptable_mass",
        "ambiguous_logical_mass",
        "hyperedge_mass",
    }
    assert result["observable_hash"]
    assert result["label_hash"]


def test_config_changed_failure_root_is_a_typed_conflict(tiny_spec, tmp_path: Path) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)[:1]
    invalid_bounds = dict(tiny_spec.component_bounds)
    invalid_bounds["repetition_data"] = (0.6, 0.7)
    failed = replace(tiny_spec, component_bounds=invalid_bounds)
    assert generate_matrix(failed, jobs, tmp_path, workers=1).failures
    with pytest.raises(ValueError, match="run manifest profile or configuration mismatch"):
        generate_matrix(tiny_spec, jobs, tmp_path, workers=1)


def test_resume_rejects_manifest_pair_id_that_disagrees_with_verified_lanes(
    tiny_spec, tmp_path: Path
) -> None:
    jobs = expand_jobs(tiny_spec, include_sealed=False)[:1]
    generate_matrix(tiny_spec, jobs, tmp_path, workers=1)
    manifest_path = tmp_path / "run_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["results"][0]["pair_id"] = "0" * 64
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    resumed = generate_matrix(tiny_spec, jobs, tmp_path, workers=1)
    assert resumed.completed == 0
    assert resumed.failures[0].code.value == "artifact_conflict"


def test_generate_pilot_dry_run_reports_profile_matrix_and_reserve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "generate-pilot",
                "--config",
                "configs/poc_pilot.json",
                "--output-root",
                str(tmp_path / "pilot"),
                "--dry-run",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["dataset_profile"] == "pilot"
    assert status["scientific_status"] == "PILOT_NOT_FINAL"
    assert status["total_jobs"] == 88
    assert status["nonsealed_jobs"] == 64
    assert status["sealed_jobs"] == 24
    assert status["required_storage_gib"] == 80
    assert set(status["allocation"]) == {"normal", "development", "sealed"}
    assert len(status["allocation"]["normal"]) == 40
    assert len(status["allocation"]["development"]) == 24
    assert len(status["allocation"]["sealed"]) == 24
    assert status["allocation"]["normal"][0] == {
        "condition_id": "repetition_d3__f01",
        "trajectory_id": 0,
        "split": "train",
    }
    assert not (tmp_path / "pilot").exists()


def test_pilot_manifest_binds_profile_geometry_and_all_expected_job_keys() -> None:
    pilot = load_spec(Path("configs/poc_pilot.json"))
    manifest = _manifest_payload({}, pilot)
    assert manifest["dataset_profile"] == "pilot"
    assert manifest["generation"] == {
        "trajectories_per_condition": 64,
        "burn_in_rounds": 4096,
        "scored_rounds": 8192,
        "mode": "standard",
        "chunk_rounds": None,
        "generation_law_version": "standard_monolithic_v1",
        "sampling_law_version": "fault_v1",
    }
    assert len(manifest["expected_job_keys"]) == 88


def test_profile_mismatched_manifest_is_a_typed_conflict(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"dataset_profile": "production", "resolved_config_hash": "different"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run manifest profile or configuration mismatch"):
        assert_run_manifest_identity(tmp_path, load_spec(Path("configs/poc_pilot.json")))


@pytest.mark.parametrize(
    "field", ["generation", "expected_job_keys", "missing_generation", "missing_expected_job_keys"]
)
def test_pilot_manifest_rejects_missing_or_altered_geometry_and_expected_job_keys(
    tmp_path: Path, field: str
) -> None:
    pilot = load_spec(Path("configs/poc_pilot.json"))
    manifest = _manifest_payload({}, pilot)
    if field == "generation":
        manifest["generation"] = {
            "trajectories_per_condition": 64,
            "burn_in_rounds": 4096,
            "scored_rounds": 65536,
        }
    elif field == "expected_job_keys":
        manifest["expected_job_keys"] = manifest["expected_job_keys"][1:]
    elif field == "missing_generation":
        del manifest["generation"]
    else:
        del manifest["expected_job_keys"]
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot manifest"):
        assert_run_manifest_identity(tmp_path, pilot)
