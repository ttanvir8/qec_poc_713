from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "kaggle_pilot_runner.py"
GUIDE = ROOT / "kaggle_run_how_to.md"
OPTIONAL_NOTEBOOK = ROOT / "kaggle_pilot_runner.ipynb"


def _read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing required Kaggle asset: {path.name}")
    return path.read_text(encoding="utf-8")


def _assert_contains(text: str, required: tuple[str, ...], *, source: str) -> None:
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"{source} missing required content: {missing}")


def _assert_no_secret_literals(text: str, *, source: str) -> None:
    forbidden = (
        r'"root_seed"',
        r"'root_seed'",
        r"public_root_seed",
        r"\b713\b",
        r"KAGGLE_KEY\s*=\s*['\"]",
        r"-----BEGIN",
    )
    for pattern in forbidden:
        if re.search(pattern, text):
            raise AssertionError(f"{source} contains a forbidden secret or raw-seed literal")


def _module_tree(text: str) -> ast.Module:
    return ast.parse(text, filename=str(RUNNER))


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{RUNNER.name} missing required function: {name}")


def _constant_assign(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    raise AssertionError(f"{RUNNER.name} missing required constant: {name}")


def _load_runner_function(
    tree: ast.Module,
    name: str,
    *,
    constants: tuple[str, ...] = (),
) -> Callable[..., object]:
    namespace: dict[str, object] = {"Path": Path, "re": re}
    nodes: list[ast.stmt] = [_constant_assign(tree, constant) for constant in constants]
    nodes.append(_function_node(tree, name))
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(RUNNER), "exec"), namespace)  # noqa: S102
    function = namespace[name]
    if not callable(function):
        raise TypeError(f"{RUNNER.name}.{name} is not callable")
    return function


def _assert_default_checkpoint_resolution(tree: ast.Module) -> None:
    resolver = _load_runner_function(
        tree,
        "resolve_checkpoint_input",
        constants=("DEFAULT_CHECKPOINT_INPUT",),
    )
    expected = Path("/kaggle/input/causaldem-pilot-checkpoint-00")
    for raw in (None, "", "   "):
        if resolver(raw) != expected:
            raise AssertionError("empty checkpoint input must resolve to the default Kaggle path")
    custom = Path("/kaggle/input/custom-checkpoint")
    if resolver(str(custom)) != custom:
        raise AssertionError("custom checkpoint input was not preserved")


def _assert_external_checkpoint_identity(tree: ast.Module) -> None:
    resolver = _load_runner_function(tree, "require_checkpoint_dataset_identity")
    for raw in (None, "", "   "):
        try:
            resolver(raw)
        except RuntimeError:
            pass
        else:
            raise AssertionError("checkpoint identity must be supplied externally")
    identity = "owner/causaldem-pilot-checkpoint"
    if resolver(f"  {identity}  ") != identity:
        raise AssertionError("external checkpoint identity was not preserved")
    version_resolver = _load_runner_function(tree, "require_checkpoint_version")
    version = f"{identity}@7"
    if version_resolver(f" {version} ", identity) != version:
        raise AssertionError("attached checkpoint version was not preserved separately")
    try:
        version_resolver("other-owner/other-checkpoint@7", identity)
    except RuntimeError:
        pass
    else:
        raise AssertionError("checkpoint version from another dataset was accepted")


def _assert_storage_projection_reserves_new_pair_and_export(tree: ast.Module) -> None:
    projection = _load_runner_function(
        tree,
        "checkpoint_storage_projection",
        constants=(
            "MAX_AUTO_SAVED_WORKING_GIB",
            "WORKING_STORAGE_STOP_GIB",
            "MIN_FREE_GIB",
            "NEW_PAIR_RESERVE_GIB",
        ),
    )
    fits = projection(1.0, 5.0)
    if fits != {
        "projected_working_with_pair_gib": 8.0,
        "projected_with_export_gib": 15.0,
        "new_pair_reserve_gib": 2,
        "fits": True,
    }:
        raise AssertionError(f"unexpected checkpoint storage projection: {fits}")
    blocked = projection(1.0, 7.0)
    if blocked["projected_working_with_pair_gib"] != 10.0:
        raise AssertionError("working-root projection omits the new pair reserve")
    if blocked["projected_with_export_gib"] != 19.0 or blocked["fits"] is not False:
        raise AssertionError("export projection omits the new pair or fails to stop")


def _assert_no_project_imports_in_notebook_interpreter(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("causaldem_qec"):
            raise AssertionError(
                "project imports must run under uv run python, not the notebook interpreter"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("causaldem_qec"):
                    raise AssertionError(
                        "project imports must run under uv run python, not the notebook interpreter"
                    )


def _assert_exact_export_allowlist(tree: ast.Module) -> None:
    allow = _load_runner_function(
        tree,
        "is_allowed_checkpoint_payload_path",
        constants=("DATA_ARTIFACT_FILES", "DATA_ARTIFACT_LANES", "DATA_ARTIFACT_SPLITS"),
    )
    allowed = (
        Path("pilot/run_manifest.json"),
        Path("pilot/data/manifests/sealed_commitment.json"),
        Path("pilot/data/observable/train/repetition_d3__f01/0/arrays.npz"),
        Path("pilot/data/labels/sealed_test/surface_d5__f14_negative/63/SHA256SUMS"),
    )
    denied = (
        Path("pilot/data/manifests/sealed_private.json"),
        Path("pilot/source/pyproject.toml"),
        Path("pilot/data/observable/train/repetition_d3__f01/0/debug.log"),
        Path("pilot/data/observable/train/repetition_d3__f01/staging.tmp/arrays.npz"),
        Path("dataset-metadata.json"),
    )
    for path in allowed:
        if allow(path) is not True:
            raise AssertionError(f"checkpoint payload allowlist rejected {path}")
    for path in denied:
        if allow(path) is not False:
            raise AssertionError(f"checkpoint payload allowlist accepted {path}")


def _assert_required_runner_structure(runner: str, tree: ast.Module) -> None:
    _assert_default_checkpoint_resolution(tree)
    _assert_external_checkpoint_identity(tree)
    _assert_storage_projection_reserves_new_pair_and_export(tree)
    _assert_no_project_imports_in_notebook_interpreter(tree)
    _assert_exact_export_allowlist(tree)
    required = (
        "run_uv_python",
        "assert_checkpoint_copy_fits",
        "delete_session_sealed_manifest",
        "finally:",
        "ensure_kaggle_cli",
        "kaggle --version",
        "git_validation",
        "20 GiB Kaggle working limit",
        "CAUSALDEM_CHECKPOINT_VERSION",
        "CAUSALDEM_CHECKPOINT_DATASET_SLUG",
        "--checkpoint-identity",
        "--checkpoint-version",
    )
    _assert_contains(runner, required, source=RUNNER.name)
    forbidden = ("forbidden_names", "commit_from_copy")
    present = [item for item in forbidden if item in runner]
    if present:
        raise AssertionError(f"{RUNNER.name} contains obsolete blocker-prone code: {present}")


def validate() -> None:
    runner = _read(RUNNER)
    guide = _read(GUIDE)
    tree = _module_tree(runner)

    required_runner_terms = (
        "/kaggle/input/causaldem-poc-source",
        "/kaggle/working/runs/pilot",
        "/kaggle/working/export/pilot",
        "uv sync --frozen --extra dev",
        "upgrade_legacy_kaggle_bootstrap",
        "verify-dataset",
        "generate-pilot",
        "--config",
        "configs/poc_pilot.json",
        "--execution-backend",
        "kaggle",
        "--job-limit",
        "1",
        "--checkpoint-root",
        "kaggle datasets version",
        "kaggle --version",
        "UserSecretsClient",
        "COMMIT_SHA.txt",
        "CAUSALDEM_CHECKPOINT_VERSION",
        "CAUSALDEM_CHECKPOINT_DATASET_SLUG",
        "--checkpoint-identity",
        "--checkpoint-version",
    )
    _assert_contains(runner, required_runner_terms, source=RUNNER.name)
    _assert_required_runner_structure(runner, tree)

    required_guide_terms = (
        "Kaggle free CPU",
        "20 GiB auto-saved",
        "private Kaggle Dataset",
        "private source dataset",
        "private checkpoint dataset",
        "uv sync --frozen --extra dev",
        "source commit",
        "legacy bootstrap",
        "44-pair checkpoint",
        "causaldem-poc generate-pilot",
        "--config configs/poc_pilot.json",
        "--execution-backend kaggle",
        "--job-limit 1",
        "--checkpoint-root",
        "18 GiB",
        "12-hour",
        "Kaggle Secret",
        "sealed manifest",
        "deleted in a finally block",
        "exact allowlist",
        "sharded checkpoint",
        "kaggle --version",
        "exact attached checkpoint dataset version",
        "stable dataset identity",
        "checkpoint_input_version",
        "2 GiB per new pair",
        "final local verification",
        "Do not paste",
    )
    _assert_contains(guide, required_guide_terms, source=GUIDE.name)

    combined = runner + "\n" + guide
    _assert_no_secret_literals(combined, source="Kaggle assets")

    if OPTIONAL_NOTEBOOK.exists():
        json.loads(OPTIONAL_NOTEBOOK.read_text(encoding="utf-8"))


if __name__ == "__main__":
    validate()
