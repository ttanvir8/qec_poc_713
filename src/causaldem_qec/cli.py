from __future__ import annotations

import argparse
import json
import os
import secrets
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from causaldem_qec.artifacts import load_sealed_seed, write_sealed_commitment
from causaldem_qec.core import PocSpec, TrajectoryJob, expand_jobs, load_spec
from causaldem_qec.simulate import generate_matrix, verify_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causaldem-poc")
    parser.add_argument("stage", choices=("freeze-sealed", "smoke", "generate", "verify-dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sealed-manifest", type=Path)
    return parser


def _sealed_commitment_path(spec: PocSpec, output_root: Path) -> Path:
    value = spec.raw["sealed_commitment_path"]
    if not isinstance(value, str):
        raise TypeError("invalid sealed commitment path")
    return output_root / value


def _sealed_jobs(spec: PocSpec, private_path: Path, output_root: Path) -> tuple[TrajectoryJob, ...]:
    commitment = _sealed_commitment_path(spec, output_root)
    seed = load_sealed_seed(private_path, commitment, purpose="sealed_evaluation")
    return tuple(
        replace(job, root_seed=seed) if job.split == "sealed_test" else job
        for job in expand_jobs(spec, include_sealed=True)
    )


def _freeze_sealed(spec: PocSpec, private_path: Path, output_root: Path) -> str:
    commitment_path = _sealed_commitment_path(spec, output_root)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(private_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    {"root_seed": int.from_bytes(secrets.token_bytes(32), "big")},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        return write_sealed_commitment(private_path, commitment_path)
    except BaseException:
        try:
            private_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _existing_jobs(spec: PocSpec, output_root: Path) -> tuple[PocSpec, tuple[TrajectoryJob, ...]]:
    manifest_path = output_root / "run_manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise TypeError("invalid run manifest")
    generation = document.get("generation")
    if not isinstance(generation, dict):
        raise TypeError("run manifest lacks generation identity")
    trajectories = generation.get("trajectories_per_condition")
    burn_in = generation.get("burn_in_rounds")
    scored = generation.get("scored_rounds")
    if (
        not isinstance(trajectories, int)
        or not isinstance(burn_in, int)
        or not isinstance(scored, int)
    ):
        raise TypeError("invalid generation identity")
    verified_spec = replace(
        spec,
        trajectories_per_condition=trajectories,
        burn_in_rounds=burn_in,
        scored_rounds=scored,
    )
    keys = {
        (item.get("condition_id"), item.get("trajectory_id"))
        for item in document["results"]
        if isinstance(item, dict)
    }
    return verified_spec, tuple(
        job
        for job in expand_jobs(verified_spec, include_sealed=False)
        if (job.condition_id, job.trajectory_id) in keys
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.config)
    if args.stage == "freeze-sealed":
        if args.sealed_manifest is None:
            raise ValueError("freeze-sealed requires --sealed-manifest")
        print(_freeze_sealed(spec, args.sealed_manifest, args.output_root))
        return 0
    if args.stage == "smoke":
        smoke = replace(spec, trajectories_per_condition=2, burn_in_rounds=128, scored_rounds=1024)
        manifest = generate_matrix(
            smoke, expand_jobs(smoke, include_sealed=False), args.output_root, workers=args.workers
        )
        print(manifest.manifest_hash)
        return 0
    if args.stage == "generate":
        jobs = (
            _sealed_jobs(spec, args.sealed_manifest, args.output_root)
            if args.sealed_manifest is not None
            else expand_jobs(spec, include_sealed=False)
        )
        manifest = generate_matrix(spec, jobs, args.output_root, workers=args.workers)
        print(manifest.manifest_hash)
        return 0
    verified_spec, jobs = _existing_jobs(spec, args.output_root)
    gates = verify_dataset(verified_spec, jobs, args.output_root)
    print(json.dumps([{"gate_id": item.gate_id, "status": item.status.value} for item in gates]))
    return 0
