"""End-to-end data pipeline: download -> reformat -> split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_triage.config import load_config
from ticket_triage.data.download import download
from ticket_triage.data.reformat import reformat
from ticket_triage.data.split import split
from ticket_triage.utils.seed import set_seed


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    raw = Path(cfg["paths"]["raw_dir"]) / "bitext.parquet"
    if not raw.exists():
        download(config_path)
    reformat(config_path)
    split(config_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    main(p.parse_args().config)
