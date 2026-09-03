"""Kaggle notebook runner for the CausalDEM-QEC pilot checkpoint loop.

Paste this file into a private Kaggle CPU notebook as cells, or upload it with
the private source dataset and run it with `python kaggle_pilot_runner.py`.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# %% [markdown]
# # CausalDEM-QEC Kaggle Pilot Runner
#
# Attach these private Kaggle Datasets before running:
#
# - `causaldem-poc-source`, containing the public source tree and COMMIT_SHA.txt.
# - The newest private pilot checkpoint dataset, containing `pilot/run_manifest.json`.
#
# Keep Internet on only while installing dependencies and publishing the private
# checkpoint dataset version.

# %%
SOURCE_INPUT = Path("/kaggle/input/causaldem-poc-source")
DEFAULT_CHECKPOINT_INPUT = Path("/kaggle/input/causaldem-pilot-checkpoint-00")


def resolve_checkpoint_input(raw_value: str | None) -> Path:
    value = (raw_value or "").strip()
    if not value:
        return DEFAULT_CHECKPOINT_INPUT
    return Path(value)


CHECKPOINT_INPUT = resolve_checkpoint_input(os.environ.get("CAUSALDEM_CHECKPOINT_INPUT"))


def require_checkpoint_dataset_identity(raw_value: str | None) -> str:
    value = (raw_value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) is None:
        raise RuntimeError("set CAUSALDEM_CHECKPOINT_DATASET_SLUG to the stable owner/dataset slug")
    return value


def require_checkpoint_version(raw_value: str | None, checkpoint_identity: str) -> str:
    value = (raw_value or "").strip()
    if re.fullmatch(re.escape(checkpoint_identity) + r"@[1-9][0-9]*", value) is None:
        raise RuntimeError(
            "set CAUSALDEM_CHECKPOINT_VERSION to the exact attached version of "
            f"{checkpoint_identity} in owner/dataset@number form"
        )
    return value


CHECKPOINT_IDENTITY = require_checkpoint_dataset_identity(
    os.environ.get("CAUSALDEM_CHECKPOINT_DATASET_SLUG")
)
CHECKPOINT_VERSION = require_checkpoint_version(
    os.environ.get("CAUSALDEM_CHECKPOINT_VERSION"), CHECKPOINT_IDENTITY
)

WORKING_SOURCE = Path("/kaggle/working/source")
WORKING_ROOT = Path("/kaggle/working/runs/pilot")
EXPORT_ROOT = Path("/kaggle/working/export")
EXPORT_PILOT = EXPORT_ROOT / "pilot"
# Expected checkpoint-root literal for review: /kaggle/working/export/pilot
SECRET_DIR = Path("/kaggle/working/secrets")
SEALED_MANIFEST_PATH = SECRET_DIR / "causaldem_pilot_sealed.json"

CONFIG = Path("configs/poc_pilot.json")
JOB_LIMIT = "1"
MAX_AUTO_SAVED_WORKING_GIB = 20
WORKING_STORAGE_STOP_GIB = 18
MIN_FREE_GIB = 2
NEW_PAIR_RESERVE_GIB = 2
SESSION_LIMIT_SECONDS = 12 * 60 * 60
STOP_BEFORE_LIMIT_SECONDS = 45 * 60
DATA_ARTIFACT_FILES = frozenset(("arrays.npz", "metadata.json", "SHA256SUMS"))
DATA_ARTIFACT_LANES = frozenset(("observable", "labels"))
DATA_ARTIFACT_SPLITS = frozenset(("train", "validation", "id_test", "development", "sealed_test"))


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout)
    return completed.stdout


def run_uv_python(source: str) -> str:
    return run(["uv", "run", "python", "-c", source], cwd=REPO)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("cannot read MemAvailable from /proc/meminfo")


def disk_report(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gib": usage.total / 1024**3,
        "used_gib": usage.used / 1024**3,
        "free_gib": usage.free / 1024**3,
    }


def assert_runtime_budget(started_at: float) -> None:
    disk = disk_report(Path("/kaggle/working"))
    elapsed = time.monotonic() - started_at
    remaining = SESSION_LIMIT_SECONDS - elapsed
    print(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "disk": disk,
                "mem_available_gib": mem_available_gib(),
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": remaining,
            },
            indent=2,
        )
    )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Kaggle runtime must use Python 3.11 for the locked environment")
    if disk["used_gib"] >= WORKING_STORAGE_STOP_GIB:
        raise RuntimeError("working storage is at or above the 18 GiB stop threshold")
    if disk["free_gib"] < MIN_FREE_GIB:
        raise RuntimeError("working storage has less than the minimum free-space reserve")
    if remaining < STOP_BEFORE_LIMIT_SECONDS:
        raise RuntimeError("too close to the 12-hour Kaggle session limit for another job")


def tree_size_gib(root: Path) -> float:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"checkpoint contains a symlink, refusing to copy: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total / 1024**3


def checkpoint_storage_projection(used_gib: float, checkpoint_gib: float) -> dict[str, object]:
    projected_working_with_pair = used_gib + checkpoint_gib + NEW_PAIR_RESERVE_GIB
    projected_with_export = projected_working_with_pair + checkpoint_gib + NEW_PAIR_RESERVE_GIB
    return {
        "projected_working_with_pair_gib": projected_working_with_pair,
        "projected_with_export_gib": projected_with_export,
        "new_pair_reserve_gib": NEW_PAIR_RESERVE_GIB,
        "fits": projected_working_with_pair < WORKING_STORAGE_STOP_GIB
        and projected_with_export <= MAX_AUTO_SAVED_WORKING_GIB - MIN_FREE_GIB,
    }


def assert_checkpoint_copy_fits(checkpoint_root: Path) -> None:
    checkpoint_gib = tree_size_gib(checkpoint_root)
    disk = disk_report(Path("/kaggle/working"))
    projection = checkpoint_storage_projection(disk["used_gib"], checkpoint_gib)
    if projection["fits"] is not True:
        raise RuntimeError(
            "current working files, checkpoint copy, 2 GiB per new pair, and full export "
            "copy would exceed the 20 GiB Kaggle working limit; "
            "attach a smaller sharded checkpoint dataset or run the checkpoint "
            "export on persistent storage before attempting this session"
        )
    print(
        json.dumps(
            {
                "checkpoint_input_gib": checkpoint_gib,
                **projection,
            },
            indent=2,
        )
    )


# %%
started_at = time.monotonic()
assert_runtime_budget(started_at)

# %%
run([sys.executable, "-m", "pip", "install", "-q", "uv"])


def locate_source_root(source_input: Path) -> Path:
    candidates = [source_input, *[path for path in source_input.iterdir() if path.is_dir()]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "uv.lock").is_file():
            return candidate
    raise RuntimeError("source dataset must contain pyproject.toml and uv.lock")


def read_source_commit_marker(source_root: Path) -> str:
    marker = source_root / "COMMIT_SHA.txt"
    if not marker.is_file():
        raise RuntimeError("source dataset must contain COMMIT_SHA.txt")
    source_commit = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{7,40}", source_commit):
        raise RuntimeError("COMMIT_SHA.txt must contain a Git commit SHA")
    return source_commit


def validate_source_commit(source_root: Path) -> tuple[str, str]:
    source_commit = read_source_commit_marker(source_root)
    if not (source_root / ".git").exists():
        return source_commit, "not_available_input_dataset_has_no_git"
    git_commit = run(["git", "rev-parse", "HEAD"], cwd=source_root).strip()
    if git_commit != source_commit:
        raise RuntimeError("source commit does not match COMMIT_SHA.txt")
    return source_commit, "matched_git_head"


def copy_source() -> Path:
    source_root = locate_source_root(SOURCE_INPUT)
    source_commit, git_validation = validate_source_commit(source_root)
    if WORKING_SOURCE.exists():
        raise RuntimeError(f"{WORKING_SOURCE} already exists; inspect before reuse")
    shutil.copytree(
        source_root,
        WORKING_SOURCE,
        ignore=shutil.ignore_patterns(
            ".superpowers",
            ".worktrees",
            "runs",
            "reports",
            "data",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ),
    )
    print(json.dumps({"source_commit": source_commit, "git_validation": git_validation}))
    return WORKING_SOURCE


REPO = copy_source()

# %%
# Keep this exact command text for notebook review: uv sync --frozen --extra dev
run(["uv", "sync", "--frozen", "--extra", "dev"], cwd=REPO)
run(
    [
        "uv",
        "run",
        "python",
        "-c",
        (
            "import importlib.metadata as m; "
            "print({name: m.version(name) for name in "
            "('numpy','scipy','stim','pymatching','pyarrow')})"
        ),
    ],
    cwd=REPO,
)


# %%
def locate_checkpoint_root(checkpoint_input: Path) -> Path:
    candidates = [
        checkpoint_input / "pilot",
        checkpoint_input,
        *[path / "pilot" for path in checkpoint_input.iterdir() if path.is_dir()],
    ]
    for candidate in candidates:
        if (candidate / "run_manifest.json").is_file():
            return candidate
    raise RuntimeError("checkpoint dataset must contain pilot/run_manifest.json")


def copy_checkpoint() -> None:
    checkpoint_root = locate_checkpoint_root(CHECKPOINT_INPUT)
    if WORKING_ROOT.exists():
        raise RuntimeError(f"{WORKING_ROOT} already exists; inspect before reuse")
    assert_checkpoint_copy_fits(checkpoint_root)
    WORKING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_root, WORKING_ROOT)
    print(
        json.dumps({"copied_checkpoint": str(checkpoint_root), "working_root": str(WORKING_ROOT)})
    )


copy_checkpoint()
assert_runtime_budget(started_at)


def upgrade_legacy_bootstrap_if_needed() -> None:
    script = f"""
import json
from pathlib import Path

from causaldem_qec.artifacts import (
    ArtifactConflict,
    inventory_checkpoint,
    upgrade_legacy_kaggle_bootstrap,
)
from causaldem_qec.core import ManifestProvenance, load_spec
from causaldem_qec.simulate import STANDARD_GENERATION_LAW_VERSION, _resolved_config_hash

repo = Path({str(REPO)!r})
config = Path({str(CONFIG)!r})
working_root = Path({str(WORKING_ROOT)!r})
checkpoint_identity = {CHECKPOINT_IDENTITY!r}
checkpoint_version = {CHECKPOINT_VERSION!r}

source_commit = (repo / "COMMIT_SHA.txt").read_text(encoding="utf-8").strip()
expected_provenance = ManifestProvenance(
    source_commit=source_commit,
    execution_backend="kaggle",
    generation_law_version=STANDARD_GENERATION_LAW_VERSION,
    checkpoint_identity=checkpoint_identity,
)
expected_hash = _resolved_config_hash(load_spec(repo / config))
manifest_path = working_root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if "provenance" in manifest:
    inventory_checkpoint(
        working_root,
        expected_config_hash=expected_hash,
        expected_provenance=expected_provenance,
        expected_checkpoint_version=checkpoint_version,
    )
    print("checkpoint already provenance-bound")
else:
    try:
        upgrade_legacy_kaggle_bootstrap(
            working_root,
            expected_config_hash=expected_hash,
            expected_provenance=expected_provenance,
            expected_checkpoint_version=checkpoint_version,
        )
    except ArtifactConflict as error:
        raise RuntimeError("legacy bootstrap manifest upgrade failed") from error
    print("legacy bootstrap manifest upgraded")
"""
    run_uv_python(script)


upgrade_legacy_bootstrap_if_needed()

# %%
run(
    [
        "uv",
        "run",
        "causaldem-poc",
        "verify-dataset",
        "--config",
        str(CONFIG),
        "--output-root",
        str(WORKING_ROOT),
    ],
    cwd=REPO,
)
assert_runtime_budget(started_at)


# %%
def write_sealed_manifest_from_secret() -> list[str]:
    use_sealed = os.environ.get("CAUSALDEM_USE_SEALED_MANIFEST", "").lower() in {"1", "true", "yes"}
    if not use_sealed:
        return []
    from kaggle_secrets import UserSecretsClient

    secret_value = UserSecretsClient().get_secret("CAUSALDEM_PILOT_SEALED_MANIFEST")
    if not secret_value:
        raise RuntimeError("sealed manifest Kaggle Secret is empty")
    SECRET_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(SEALED_MANIFEST_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(secret_value)
        handle.flush()
        os.fsync(handle.fileno())
    return ["--sealed-manifest", str(SEALED_MANIFEST_PATH)]


def delete_session_sealed_manifest() -> None:
    if SEALED_MANIFEST_PATH.exists():
        SEALED_MANIFEST_PATH.unlink()
        print(json.dumps({"sealed_manifest_deleted": str(SEALED_MANIFEST_PATH)}))
    try:
        SECRET_DIR.rmdir()
    except OSError:
        pass


# %%
try:
    sealed_args = write_sealed_manifest_from_secret()
    generate_command = [
        "uv",
        "run",
        "causaldem-poc",
        "generate-pilot",
        "--config",
        str(CONFIG),
        "--output-root",
        str(WORKING_ROOT),
        "--workers",
        "1",
        "--execution-backend",
        "kaggle",
        "--job-limit",
        JOB_LIMIT,
        "--checkpoint-root",
        str(EXPORT_PILOT),
        "--checkpoint-identity",
        CHECKPOINT_IDENTITY,
        "--checkpoint-version",
        CHECKPOINT_VERSION,
        *sealed_args,
    ]
    run(generate_command, cwd=REPO)
    assert_runtime_budget(started_at)

    # %%
    run(
        [
            "uv",
            "run",
            "causaldem-poc",
            "verify-dataset",
            "--config",
            str(CONFIG),
            "--output-root",
            str(WORKING_ROOT),
        ],
        cwd=REPO,
    )
finally:
    delete_session_sealed_manifest()


# %%
def is_allowed_checkpoint_payload_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    if parts == ("pilot", "run_manifest.json"):
        return True
    if parts == ("pilot", "data", "manifests", "sealed_commitment.json"):
        return True
    if len(parts) != 7:
        return False
    root, data, lane, split, condition_id, trajectory_id, filename = parts
    return (
        root == "pilot"
        and data == "data"
        and lane in DATA_ARTIFACT_LANES
        and split in DATA_ARTIFACT_SPLITS
        and bool(condition_id)
        and not condition_id.startswith(".")
        and trajectory_id.isdecimal()
        and filename in DATA_ARTIFACT_FILES
    )


def assert_export_is_uploadable() -> None:
    if not (EXPORT_PILOT / "run_manifest.json").is_file():
        raise RuntimeError("checkpoint export was not produced; inspect generation status")
    for path in EXPORT_ROOT.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink in export staging: {path}")
        if path.is_file() and not is_allowed_checkpoint_payload_path(path.relative_to(EXPORT_ROOT)):
            raise RuntimeError(f"path is outside the exact checkpoint payload allowlist: {path}")
    manifest = json.loads((EXPORT_PILOT / "run_manifest.json").read_text(encoding="utf-8"))
    completed = sum(item.get("completed") is True for item in manifest["results"])
    print(json.dumps({"export_root": str(EXPORT_ROOT), "completed_pairs": completed}))


assert_export_is_uploadable()


# %%
def ensure_kaggle_cli() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        run(["uv", "tool", "install", "kaggle"])
        local_bin = Path.home() / ".local" / "bin"
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("Kaggle CLI is unavailable after uv tool install kaggle")
    # Capability check: kaggle --version
    run([executable, "--version"])
    return executable


def configure_kaggle_credentials() -> None:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = client.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = client.get_secret("KAGGLE_KEY")
    if not os.environ["KAGGLE_USERNAME"] or not os.environ["KAGGLE_KEY"]:
        raise RuntimeError("Kaggle API credentials are missing")


def write_dataset_metadata() -> tuple[str, str]:
    owner_slug = CHECKPOINT_IDENTITY
    owner, slug = owner_slug.split("/", 1)
    metadata = {
        "id": f"{owner}/{slug}",
        "title": "CausalDEM-QEC pilot checkpoint",
        "licenses": [{"name": "CC0-1.0"}],
    }
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (EXPORT_ROOT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return owner_slug, slug


def publish_private_checkpoint() -> None:
    kaggle = ensure_kaggle_cli()
    configure_kaggle_credentials()
    owner_slug, _slug = write_dataset_metadata()
    manifest = json.loads((EXPORT_PILOT / "run_manifest.json").read_text(encoding="utf-8"))
    completed = sum(item.get("completed") is True for item in manifest["results"])
    source_commit = (REPO / "COMMIT_SHA.txt").read_text(encoding="utf-8").strip()
    message = f"pilot checkpoint: completed {completed} of 88; source {source_commit}"
    create_first = os.environ.get("CAUSALDEM_CREATE_CHECKPOINT_DATASET", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if create_first:
        run(
            [
                kaggle,
                "datasets",
                "create",
                "-p",
                str(EXPORT_ROOT),
                "--dir-mode",
                "zip",
            ]
        )
    else:
        # Kaggle private checkpoint upload command: kaggle datasets version
        run(
            [
                kaggle,
                "datasets",
                "version",
                "-p",
                str(EXPORT_ROOT),
                "-m",
                message,
                "--dir-mode",
                "zip",
            ]
        )
    print(json.dumps({"private_dataset": owner_slug, "version_message": message}))


publish_private_checkpoint()
