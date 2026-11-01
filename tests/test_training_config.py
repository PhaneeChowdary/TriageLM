import os
from unittest.mock import patch

import pytest

from ticket_triage.config import load_training_config


@pytest.fixture
def files(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("output:\n  push_to_hub: false\n  hub_repo_id: null\n  adapter_dir: local\n")
    return config, tmp_path / ".env"


def test_local_defaults_without_env(files):
    config, env = files
    with patch.dict(os.environ, {}, clear=True):
        output = load_training_config(config, env)["output"]
    assert output == {"push_to_hub": False, "hub_repo_id": None, "adapter_dir": "local"}


def test_private_file_enables_upload(files):
    config, env = files
    env.write_text("TRIAGE_PUSH_TO_HUB=true\nTRIAGE_HUB_REPO_ID=test-owner/test-repo\n")
    with patch.dict(os.environ, {}, clear=True):
        output = load_training_config(config, env)["output"]
    assert output["push_to_hub"] is True
    assert output["hub_repo_id"] == "test-owner/test-repo"
    assert output["adapter_dir"] == "local"


def test_shell_overrides_private_file(files):
    config, env = files
    env.write_text("TRIAGE_PUSH_TO_HUB=true\nTRIAGE_HUB_REPO_ID=test-owner/test-repo\n")
    with patch.dict(os.environ, {"TRIAGE_PUSH_TO_HUB": "false"}, clear=True):
        assert load_training_config(config, env)["output"]["push_to_hub"] is False


@pytest.mark.parametrize("settings", [
    {"TRIAGE_PUSH_TO_HUB": "maybe"},
    {"TRIAGE_PUSH_TO_HUB": "true"},
    {"TRIAGE_PUSH_TO_HUB": "true", "TRIAGE_HUB_REPO_ID": "CHANGE_ME/repo"},
])
def test_bad_upload_settings_fail_before_training(files, settings):
    config, env = files
    with patch.dict(os.environ, settings, clear=True), pytest.raises(ValueError):
        load_training_config(config, env)
