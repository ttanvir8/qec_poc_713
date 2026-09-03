import os
import sys
import types

import pytest

from causaldem_qec.kaggle_auth import configure_kaggle_credentials


def test_configure_kaggle_credentials_loads_attached_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = types.SimpleNamespace(
        get_secret=lambda name: {
            "KAGGLE_USERNAME": "pilot-owner",
            "KAGGLE_KEY": "pilot-key",
        }[name]
    )
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        types.SimpleNamespace(UserSecretsClient=lambda: client),
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    configure_kaggle_credentials()

    assert os.environ["KAGGLE_USERNAME"] == "pilot-owner"
    assert os.environ["KAGGLE_KEY"] == "pilot-key"


def test_configure_kaggle_credentials_rejects_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = types.SimpleNamespace(get_secret=lambda _name: "")
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        types.SimpleNamespace(UserSecretsClient=lambda: client),
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Kaggle API credentials are missing"):
        configure_kaggle_credentials()
