"""Download the Bitext dataset to data/raw as parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from ticket_triage.config import load_config


def download(config_path: str) -> Path:
    cfg = load_config(config_path)
    src = cfg["source"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(src["hf_dataset"], split=src["split"])
    out = raw_dir / "bitext.parquet"
    ds.to_parquet(out)
    print(f"Saved {len(ds)} rows -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    download(p.parse_args().config)
