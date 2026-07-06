"""End-to-end smoke test: run the eval harness on ~50 real rows with a mock.

Proves the whole pipeline works offline, no GPU or network.
Requires `make data` to have produced data/processed/test.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

from ticket_triage.config import load_config
from ticket_triage.eval.harness import evaluate
from ticket_triage.eval.report import markdown_table
from ticket_triage.eval.robustness import corrupt_records
from ticket_triage.models.generator import MockGenerator

N = 50


def _mock(records):
    return MockGenerator({r["instruction"]: r["target"] for r in records})


def main() -> None:
    test = Path("data/processed/test.jsonl")
    if not test.exists():
        raise SystemExit("Run `make data` first to produce data/processed/test.jsonl")

    records = [json.loads(line) for _, line in zip(range(N), test.open(), strict=False)]
    eval_cfg = load_config("configs/eval.yaml")

    # One mock keyed on CLEAN tickets, evaluated on both slices: corrupted inputs
    # it can't recognize should visibly tank the metrics - that's the harness working.
    gen = _mock(records)
    ood = corrupt_records(records, eval_cfg["robustness"], eval_cfg["robustness"]["seed"])
    reports = {
        "mock (clean)": evaluate(gen, records),
        "mock (OOD)": evaluate(gen, ood),
    }
    print(markdown_table(reports))


if __name__ == "__main__":
    main()
