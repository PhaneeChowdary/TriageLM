"""Sample a fraction of the test set into a manual-review queue as JSONL.

A reviewer fills in `human_notes` / `looks_correct` against the gold labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ticket_triage.config import load_config


def sample(data_config: str, eval_config: str) -> Path:
    dcfg = load_config(data_config)
    ecfg = load_config(eval_config)
    rate = ecfg["judge"]["sample_rate"]

    test = Path(dcfg["paths"]["processed_dir"]) / "test.jsonl"
    rows = [json.loads(line) for line in test.open()]
    rng = random.Random(dcfg["seed"])
    picked = rng.sample(rows, k=max(1, round(len(rows) * rate)))

    out = Path(ecfg["reporting"]["out_dir"]) / "manual_review_sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"],
                "instruction": r["instruction"],
                "gold": r["target"],
                "looks_correct": None,
                "human_notes": "",
            }) + "\n")
    print(f"Sampled {len(picked)}/{len(rows)} -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    args = p.parse_args()
    sample(args.data_config, args.eval_config)
