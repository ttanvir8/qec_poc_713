from __future__ import annotations

import ast
import json
import re
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


def validate() -> None:
    runner = _read(RUNNER)
    guide = _read(GUIDE)
    ast.parse(runner, filename=str(RUNNER))

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
        "UserSecretsClient",
        "COMMIT_SHA.txt",
    )
    _assert_contains(runner, required_runner_terms, source=RUNNER.name)

    required_guide_terms = (
        "Kaggle free CPU",
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
