import ast
from pathlib import Path


def _load_clone_command():
    source = Path("kaggle_run/kaggle_pilot_runner.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "clone_command"
    )
    namespace = {"Path": Path}
    exec(  # noqa: S102 - execute only the extracted pure test target
        compile(ast.Module([function], type_ignores=[]), "kaggle_pilot_runner.py", "exec"),
        namespace,
    )
    return namespace["clone_command"]


def _load_should_reuse_clone():
    source = Path("kaggle_run/kaggle_pilot_runner.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "should_reuse_clone"
    )
    namespace = {"Path": Path}
    exec(  # noqa: S102 - execute only the extracted pure test target
        compile(ast.Module([function], type_ignores=[]), "kaggle_pilot_runner.py", "exec"),
        namespace,
    )
    return namespace["should_reuse_clone"]


def test_clone_command_uses_configured_public_repo_and_ref(tmp_path: Path) -> None:
    destination = tmp_path / "source"
    clone_command = _load_clone_command()

    assert clone_command(
        "https://github.com/ttanvir8/qec_poc_713.git",
        "main",
        destination,
    ) == [
        "git",
        "clone",
        "--branch",
        "main",
        "--single-branch",
        "https://github.com/ttanvir8/qec_poc_713.git",
        str(destination),
    ]


def test_existing_git_checkout_can_be_reused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / "pyproject.toml").write_text("", encoding="utf-8")

    assert _load_should_reuse_clone()(source) is True
