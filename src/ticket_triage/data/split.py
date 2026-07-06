"""Stratified train/val/test split by intent. Deterministic; test held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from ticket_triage.config import load_config


def stratified_split(
    rows: list[dict], ratios: dict, stratify_by: str, seed: int
) -> dict[str, list[dict]]:
    labels = [r["target"][stratify_by] for r in rows]
    train_val, test = train_test_split(
        rows, test_size=ratios["test"], random_state=seed, stratify=labels
    )
    val_frac = ratios["val"] / (ratios["train"] + ratios["val"])
    tv_labels = [r["target"][stratify_by] for r in train_val]
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=seed, stratify=tv_labels
    )
    return {"train": train, "val": val, "test": test}


def split(config_path: str) -> None:
    cfg = load_config(config_path)
    interim = Path(cfg["paths"]["interim_dir"]) / "reformatted.jsonl"
    rows = [json.loads(line) for line in interim.open()]

    splits = stratified_split(
        rows, cfg["split"]["ratios"], cfg["split"]["stratify_by"], cfg["seed"]
    )

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, rows_ in splits.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows_:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows_)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    split(p.parse_args().config)
