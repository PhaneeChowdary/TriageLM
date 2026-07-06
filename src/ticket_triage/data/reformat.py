"""Reformat raw Bitext into chat records with derived fields and gold entities."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ticket_triage.config import load_config
from ticket_triage.data.derive import derive
from ticket_triage.models.prompts import build_messages
from ticket_triage.schema import validate_output

PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")

PLACEHOLDER_TO_SLOT = {
    "Order Number": "order_id",
    "Account Type": "account_type",
    "Account Category": "account_category",
    "Person Name": "person_name",
    "Refund Amount": "refund_amount",
    "Currency Symbol": "refund_amount",
    "Delivery City": "delivery_city",
    "Delivery Country": "delivery_country",
    "Invoice Number": "invoice_id",
}
ENTITY_SLOTS = [
    "order_id", "account_type", "account_category", "person_name",
    "refund_amount", "delivery_city", "delivery_country", "invoice_id",
]


def extract_entities(instruction: str) -> dict:
    ents: dict = dict.fromkeys(ENTITY_SLOTS)
    for name in PLACEHOLDER.findall(instruction):
        slot = PLACEHOLDER_TO_SLOT.get(name.strip())
        if slot and ents[slot] is None:
            ents[slot] = "{{" + name.strip() + "}}"
    return ents


def to_record(row: dict, idx: int) -> dict:
    target = {
        "intent": row["intent"],
        "category": row["category"],
        "entities": extract_entities(row["instruction"]),
        **derive(row["intent"], row["category"]),
    }
    validate_output(target)  # guarantee the pipeline emits schema-valid data
    return {
        "id": idx,
        "instruction": row["instruction"],
        "target": target,
        "messages": build_messages(row["instruction"], target),
    }


def reformat(config_path: str) -> Path:
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw_dir"]) / "bitext.parquet"
    df = pd.read_parquet(raw)

    out_dir = Path(cfg["paths"]["interim_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "reformatted.jsonl"
    with out.open("w") as f:
        for i, row in enumerate(df.to_dict("records")):
            f.write(json.dumps(to_record(row, i)) + "\n")
    print(f"Reformatted {len(df)} rows -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    reformat(p.parse_args().config)
