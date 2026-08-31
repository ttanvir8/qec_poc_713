from pathlib import Path

import numpy as np
import pytest

from causaldem_qec.core import derive_seed, expand_jobs, load_spec, source_cutoff, target_interval

CONFIG = Path("configs/poc.json")


def test_config_expands_exact_committed_matrix() -> None:
    spec = load_spec(CONFIG)
    jobs = expand_jobs(spec, include_sealed=True)
    assert len({job.condition_id for job in jobs}) == 30
    assert len(jobs) == 1920
    assert sum(job.split == "train" for job in jobs) == 320
    assert sum(job.split == "validation" for job in jobs) == 160
    assert sum(job.split == "id_test" for job in jobs) == 160


def test_resolved_manifest_has_exact_trajectory_and_round_totals() -> None:
    spec = load_spec(CONFIG)
    jobs = expand_jobs(spec, include_sealed=True)
    assert sum(job.split == "development" for job in jobs) == 512
    assert sum(job.split == "sealed_test" for job in jobs) == 768
    assert len(jobs) * spec.burn_in_rounds == 7_864_320
    assert len(jobs) * spec.scored_rounds == 125_829_120
    assert len(jobs) * (spec.scored_rounds // spec.episode_rounds) == 3_932_160
    assert len(jobs) * (spec.scored_rounds // spec.block_rounds) == 491_520


def test_semantic_seed_does_not_depend_on_call_order() -> None:
    first = derive_seed(713, "surface_d3", "f03", 7, "dynamics").generate_state(8)
    derive_seed(713, "unused").generate_state(8)
    second = derive_seed(713, "surface_d3", "f03", 7, "dynamics").generate_state(8)
    np.testing.assert_array_equal(first, second)


def test_block_b_forecasts_exactly_block_b_plus_two() -> None:
    assert source_cutoff(block=4, block_rounds=256) == 1279
    assert target_interval(block=4, block_rounds=256) == (1536, 1792)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "unknown": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_spec(path)
