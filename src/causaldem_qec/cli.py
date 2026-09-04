from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from causaldem_qec.artifacts import load_sealed_seed, write_sealed_commitment
from causaldem_qec.core import (
    ExecutionOptions,
    ManifestProvenance,
    PocSpec,
    TrajectoryJob,
    expand_jobs,
    load_spec,
)
from causaldem_qec.report import build_dataset_eda
from causaldem_qec.simulate import (
    STANDARD_GENERATION_LAW_VERSION,
    GateStatus,
    assert_run_manifest_identity,
    generate_bounded_checkpoint,
    generate_matrix,
    verify_dataset,
)

_PILOT_RESERVE_GIB = 80


class StoragePreflightError(ValueError):
    """The pilot root does not have the documented minimum free space."""


PILOT_PARTITIONS = ("shard1", "shard2", "shard3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causaldem-poc")
    parser.add_argument(
        "stage",
        choices=(
            "freeze-sealed",
            "smoke",
            "generate",
            "generate-pilot",
            "verify-dataset",
            "eda-dataset",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/poc.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--execution-backend", choices=("local", "kaggle"), default="local")
    parser.add_argument("--job-limit", type=int)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-identity")
    parser.add_argument("--checkpoint-version")
    parser.add_argument(
        "--generation-mode",
        choices=("standard", "bounded"),
        default="standard",
        help="trajectory generation strategy",
    )
    parser.add_argument(
        "--generation-chunk-rounds",
        type=int,
        help="number of rounds processed per bounded-generation chunk",
    )
    parser.add_argument(
        "--pilot-partition",
        choices=PILOT_PARTITIONS,
        help="generate one fresh non-sealed pilot shard",
    )
    parser.add_argument("--sealed-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reports-root", type=Path, default=Path("reports/dataset_eda"))
    return parser


def select_pilot_partition(
    spec: PocSpec,
    jobs: Sequence[TrajectoryJob],
    partition: str | None,
) -> tuple[TrajectoryJob, ...]:
    """Select a deterministic fresh shard without changing the pilot config."""
    if partition is None:
        return tuple(jobs)
    if spec.dataset_profile != "pilot":
        raise ValueError("pilot partition selection requires the pilot configuration")
    if partition not in PILOT_PARTITIONS:
        raise ValueError(f"unknown pilot partition: {partition}")

    normal = tuple(spec.pilot_partitions["normal"])
    development = tuple(spec.pilot_partitions["development"])
    nonsealed_conditions = normal + development
    condition_groups = {
        "shard1": frozenset(nonsealed_conditions[:5]),
        "shard2": frozenset(nonsealed_conditions[5:10]),
        "shard3": frozenset(nonsealed_conditions[10:]),
    }
    selected_conditions = condition_groups[partition]
    return tuple(job for job in jobs if job.condition_id in selected_conditions)


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


def _pilot_config_path() -> Path:
    return Path("configs/poc_pilot.json").resolve()


def _require_pilot_config(config_path: Path, spec: PocSpec) -> None:
    if spec.dataset_profile != "pilot" or config_path.resolve() != _pilot_config_path():
        raise ValueError("generate-pilot requires configs/poc_pilot.json")


def _pilot_preflight(output_root: Path, spec: PocSpec) -> int:
    full_run_root = Path(spec.roots["runs"]).resolve()
    if output_root.resolve() in {Path.cwd(), full_run_root}:
        raise ValueError("pilot output root must be distinct from the full-production root")
    assert_run_manifest_identity(output_root, spec)
    probe = output_root.resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    reserve = _PILOT_RESERVE_GIB * 1024**3
    if free < reserve:
        raise StoragePreflightError(
            f"pilot storage preflight requires {_PILOT_RESERVE_GIB} GiB free, found {free} bytes"
        )
    return free


def _source_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    commit_file = repository / "COMMIT_SHA.txt"
    if commit_file.is_file():
        value = commit_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot determine source commit") from error


def _path_contains(parent: Path, child: Path) -> bool:
    parent, child = parent.resolve(), child.resolve()
    return parent == child or parent in child.parents


def _execution_context(
    args: argparse.Namespace, spec: PocSpec
) -> tuple[ExecutionOptions, ManifestProvenance | None]:
    if args.generation_mode == "bounded":
        if args.generation_chunk_rounds is None:
            raise ValueError("generation chunk rounds are required for bounded generation")
        if args.generation_chunk_rounds % spec.episode_rounds:
            raise ValueError("generation chunk rounds must be a multiple of episode_rounds")
    elif args.generation_chunk_rounds is not None:
        raise ValueError("generation chunk rounds require bounded generation mode")
    options = ExecutionOptions(
        execution_backend=args.execution_backend,
        job_limit=args.job_limit,
        checkpoint_identity=args.checkpoint_identity,
        checkpoint_version=args.checkpoint_version,
        generation_mode=args.generation_mode,
        generation_chunk_rounds=args.generation_chunk_rounds,
    )
    if options.execution_backend == "local":
        if (
            args.job_limit is not None
            or args.checkpoint_root is not None
            or args.checkpoint_identity is not None
            or args.checkpoint_version is not None
        ):
            raise ValueError(
                "--job-limit, --checkpoint-root, --checkpoint-identity, and "
                "--checkpoint-version require "
                "--execution-backend kaggle"
            )
        return options, None
    if args.stage != "generate-pilot" or spec.dataset_profile != "pilot":
        raise ValueError("kaggle execution is supported only for generate-pilot")
    if (
        options.job_limit is None
        or args.checkpoint_root is None
        or options.checkpoint_identity is None
        or options.checkpoint_version is None
    ):
        raise ValueError(
            "kaggle execution requires --job-limit, --checkpoint-root, and an externally "
            "supplied checkpoint identity and version via --checkpoint-identity and "
            "--checkpoint-version"
        )
    if args.workers != 1:
        raise ValueError("kaggle execution requires --workers 1")
    if _path_contains(args.output_root, args.checkpoint_root) or _path_contains(
        args.checkpoint_root, args.output_root
    ):
        raise ValueError("checkpoint root must be separate from the output root")
    if args.sealed_manifest is not None and (
        _path_contains(args.checkpoint_root, args.sealed_manifest)
        or _path_contains(args.output_root, args.sealed_manifest)
    ):
        raise ValueError("private sealed manifest must be outside output and checkpoint roots")
    provenance = ManifestProvenance(
        source_commit=_source_commit(),
        execution_backend="kaggle",
        generation_law_version=STANDARD_GENERATION_LAW_VERSION,
        checkpoint_identity=options.checkpoint_identity,
        generation_mode=options.generation_mode,
        generation_chunk_rounds=options.generation_chunk_rounds,
    )
    return options, provenance


def _pilot_status(
    spec: PocSpec,
    jobs: tuple[TrajectoryJob, ...],
    *,
    free_bytes: int,
    dry_run: bool,
    manifest: str | None,
    execution_options: ExecutionOptions | None = None,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    allocation: dict[str, list[dict[str, str | int]]] = {
        partition: [] for partition in ("normal", "development", "sealed")
    }
    partition_by_condition = {
        condition_id: partition
        for partition, condition_ids in spec.pilot_partitions.items()
        for condition_id in condition_ids
    }
    for job in expand_jobs(spec, include_sealed=True):
        allocation[partition_by_condition[job.condition_id]].append(
            {
                "condition_id": job.condition_id,
                "trajectory_id": job.trajectory_id,
                "split": job.split,
            }
        )
    status: dict[str, object] = {
        "dataset_profile": spec.dataset_profile,
        "scientific_status": "PILOT_NOT_FINAL",
        "total_jobs": len(expand_jobs(spec, include_sealed=True)),
        "nonsealed_jobs": len(expand_jobs(spec, include_sealed=False)),
        "sealed_jobs": sum(
            job.split == "sealed_test" for job in expand_jobs(spec, include_sealed=True)
        ),
        "sealed_access": "validated" if len(jobs) == 88 else "requires_private_manifest",
        "scheduled_jobs": len(jobs),
        "allocation": allocation,
        "required_storage_gib": _PILOT_RESERVE_GIB,
        "available_storage_bytes": free_bytes,
        "dry_run": dry_run,
        "manifest_hash": manifest,
    }
    if execution_options is not None:
        status.update(
            {
                "execution_backend": execution_options.execution_backend,
                "job_limit": execution_options.job_limit,
                "checkpoint_root": None
                if checkpoint_root is None
                else str(checkpoint_root.resolve()),
                "checkpoint_identity": execution_options.checkpoint_identity,
                "checkpoint_input_version": execution_options.checkpoint_version,
            }
        )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.config)
    execution_options, provenance = _execution_context(args, spec)
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
        jobs = select_pilot_partition(spec, jobs, args.pilot_partition)
        manifest = generate_matrix(
            spec,
            jobs,
            args.output_root,
            workers=args.workers,
            execution_options=execution_options,
            provenance=provenance,
        )
        print(manifest.manifest_hash)
        return 0
    if args.stage == "generate-pilot":
        _require_pilot_config(args.config, spec)
        if execution_options.execution_backend == "kaggle":
            assert_run_manifest_identity(
                args.output_root,
                spec,
                provenance,
                checkpoint_version=execution_options.checkpoint_version,
            )
            probe = args.output_root.resolve()
            while not probe.exists():
                probe = probe.parent
            free_bytes = shutil.disk_usage(probe).free
        else:
            free_bytes = _pilot_preflight(args.output_root, spec)
        jobs = (
            _sealed_jobs(spec, args.sealed_manifest, args.output_root)
            if args.sealed_manifest is not None
            else expand_jobs(spec, include_sealed=False)
        )
        jobs = select_pilot_partition(spec, jobs, args.pilot_partition)
        if args.dry_run:
            print(
                json.dumps(
                    _pilot_status(
                        spec,
                        jobs,
                        free_bytes=free_bytes,
                        dry_run=True,
                        manifest=None,
                        execution_options=execution_options,
                        checkpoint_root=args.checkpoint_root,
                    )
                )
            )
            return 0
        if execution_options.execution_backend == "kaggle":
            assert args.checkpoint_root is not None
            assert provenance is not None
            manifest = generate_bounded_checkpoint(
                spec,
                jobs,
                args.output_root,
                args.checkpoint_root,
                workers=args.workers,
                execution_options=execution_options,
                provenance=provenance,
            )
        else:
            manifest = generate_matrix(
                spec,
                jobs,
                args.output_root,
                workers=args.workers,
                execution_options=execution_options,
                provenance=provenance,
            )
        print(
            json.dumps(
                _pilot_status(
                    spec,
                    jobs,
                    free_bytes=free_bytes,
                    dry_run=False,
                    manifest=manifest.manifest_hash,
                    execution_options=execution_options,
                    checkpoint_root=args.checkpoint_root,
                )
            )
        )
        return 0
    if args.stage == "eda-dataset":
        _require_pilot_config(args.config, spec)
        try:
            assert_run_manifest_identity(
                args.output_root,
                spec,
                allow_bound_provenance=True,
            )
        except ValueError as error:
            raise ValueError(
                "eda-dataset requires a complete pilot root with verified artifacts"
            ) from error
        gates = verify_dataset(spec, expand_jobs(spec, include_sealed=True), args.output_root)
        if any(item.status is not GateStatus.PASS for item in gates if item.gate_id != "DQ08"):
            raise ValueError("eda-dataset requires a complete pilot root with verified artifacts")
        index = build_dataset_eda(
            args.output_root,
            args.reports_root,
            sample_seed=spec.public_root_seed,
            chunk_rounds=spec.chunk_rounds,
            gate_results=gates,
        )
        print(json.dumps({"scientific_status": "PILOT / NOT FINAL", "sections": index.sections}))
        return 0
    verified_spec, jobs = _existing_jobs(spec, args.output_root)
    gates = verify_dataset(verified_spec, jobs, args.output_root)
    print(json.dumps([{"gate_id": item.gate_id, "status": item.status.value} for item in gates]))
    return 0
