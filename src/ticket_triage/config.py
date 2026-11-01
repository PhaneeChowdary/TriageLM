"""Load a YAML run config into a dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} is not a mapping.")
    return data


def load_training_config(path: str | Path, env_path: str | Path | None = None) -> dict[str, Any]:
    """Apply private publishing settings before loading GPU libraries.

    Shell/notebook environment > project .env > YAML defaults.
    """
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env" if env_path is None else env_path, override=False)
    cfg = load_config(path)
    output = cfg["output"]
    setting = os.environ.get("TRIAGE_PUSH_TO_HUB")
    if setting is not None:
        value = setting.strip().lower()
        if value not in {"true", "false", "1", "0"}:
            raise ValueError("TRIAGE_PUSH_TO_HUB must be true, false, 1, or 0.")
        output["push_to_hub"] = value in {"true", "1"}
    if "TRIAGE_HUB_REPO_ID" in os.environ:
        output["hub_repo_id"] = os.environ["TRIAGE_HUB_REPO_ID"].strip()
    if type(output.get("push_to_hub")) is not bool:
        raise ValueError("output.push_to_hub must be a boolean.")
    if output["push_to_hub"]:
        repo = output.get("hub_repo_id")
        if (
            not isinstance(repo, str)
            or "CHANGE_ME" in repo
            or len(repo.split("/")) != 2
            or not all(repo.split("/"))
        ):
            raise ValueError("Set TRIAGE_HUB_REPO_ID to owner/repository before enabling uploads.")
    return cfg
