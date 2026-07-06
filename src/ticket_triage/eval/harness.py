"""Run a generator over records and score it. Backend-agnostic."""

from __future__ import annotations

import time
from statistics import mean

from ticket_triage.eval import metrics
from ticket_triage.eval.parse import parse
from ticket_triage.models.generator import Generator


def _field(obj: dict | None, key: str) -> str:
    value = obj.get(key) if obj else None
    return value if isinstance(value, str) else "<invalid>"


def evaluate(generator: Generator, records: list[dict]) -> dict:
    golds = [r["target"] for r in records]

    outputs, latencies = [], []
    for r in records:
        start = time.perf_counter()
        outputs.append(generator.generate(r["instruction"]))
        latencies.append(time.perf_counter() - start)

    parsed = [parse(o) for o in outputs]
    n = len(records)
    return {
        "n": n,
        "json_parse_rate": sum(p.json_ok for p in parsed) / n,
        "schema_valid_rate": sum(p.schema_ok for p in parsed) / n,
        "intent_macro_f1": metrics.intent_macro_f1(
            [_field(p.obj, "intent") for p in parsed], [g["intent"] for g in golds]
        ),
        "category_accuracy": metrics.category_accuracy(
            [_field(p.obj, "category") for p in parsed], [g["category"] for g in golds]
        ),
        "entity": metrics.entity_prf(
            [(p.obj or {}).get("entities") for p in parsed], [g["entities"] for g in golds]
        ),
        "latency_ms_mean": 1000 * mean(latencies) if latencies else 0.0,
    }
