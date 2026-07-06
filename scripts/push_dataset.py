"""Push the reformatted train/val/test splits to the HF Hub.

Runs locally and needs an HF token via `huggingface-cli login` or HF_TOKEN.
    python scripts/push_dataset.py --repo-id <your-hf-username>/ticket-triage-reformatted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from ticket_triage.config import load_config


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(repo_id: str, config_path: str) -> None:
    cfg = load_config(config_path)
    proc = Path(cfg["paths"]["processed_dir"])
    splits = DatasetDict({
        name: Dataset.from_list(_load_jsonl(proc / f"{name}.jsonl"))
        for name in ("train", "val", "test")
    })
    splits.push_to_hub(repo_id)
    print("pushed dataset ->", repo_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    args = p.parse_args()
    main(args.repo_id, args.config)
