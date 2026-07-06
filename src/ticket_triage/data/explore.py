"""Profile the raw dataset: label distributions and entity placeholders."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ticket_triage.config import load_config

PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")


def profile(config_path: str) -> dict:
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw_dir"]) / "bitext.parquet"
    df = pd.read_parquet(raw)

    intents = df["intent"].value_counts()
    categories = df["category"].value_counts()
    placeholders = Counter(m for t in df["instruction"] for m in PLACEHOLDER.findall(t))

    report = {
        "rows": len(df),
        "columns": list(df.columns),
        "n_intents": int(intents.size),
        "n_categories": int(categories.size),
        "intents": intents.to_dict(),
        "categories": categories.to_dict(),
        "entity_placeholders": dict(placeholders.most_common()),
    }

    out = Path(cfg["paths"]["interim_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_profile.json").write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _print_summary(r: dict) -> None:
    print(f"rows={r['rows']}  intents={r['n_intents']}  categories={r['n_categories']}")
    print("columns:", ", ".join(r["columns"]))
    print("\ncategories:")
    for k, v in r["categories"].items():
        print(f"  {v:6d}  {k}")
    print("\nentity placeholders:")
    for k, v in r["entity_placeholders"].items():
        print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    profile(p.parse_args().config)
