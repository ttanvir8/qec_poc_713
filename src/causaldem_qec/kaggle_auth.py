"""Load Kaggle Notebook Secrets into the environment used by the Kaggle CLI."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, cast


def configure_kaggle_credentials() -> None:
    """Load the attached Kaggle API secrets without exposing their values."""
    try:
        secrets_module = cast(Any, import_module("kaggle_secrets"))
    except ImportError as error:
        raise RuntimeError(
            "Kaggle Secrets are available only inside a Kaggle Notebook; attach "
            "KAGGLE_USERNAME and KAGGLE_KEY before publishing"
        ) from error

    user_secrets_client = cast(Any, secrets_module.UserSecretsClient)
    client = user_secrets_client()
    username = client.get_secret("KAGGLE_USERNAME")
    key = client.get_secret("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("Kaggle API credentials are missing")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
