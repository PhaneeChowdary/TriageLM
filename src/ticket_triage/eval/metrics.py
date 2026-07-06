"""Triage metrics: intent macro-F1, category accuracy, entity P/R/F.

Entity scoring is presence-aware: a gold "positive" is a non-null slot, so the
metric measures extraction rather than the model's ability to emit all-null.
"""

from __future__ import annotations

from sklearn.metrics import f1_score

from ticket_triage.schema import Entities

ENTITY_SLOTS = list(Entities.model_fields)


def intent_macro_f1(preds: list[str], golds: list[str]) -> float:
    labels = sorted(set(golds))
    return float(f1_score(golds, preds, labels=labels, average="macro", zero_division=0))


def category_accuracy(preds: list[str], golds: list[str]) -> float:
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds, strict=True)) / len(golds)


def entity_prf(preds: list[dict | None], golds: list[dict]) -> dict:
    tp = fp = fn = 0
    for pe, ge in zip(preds, golds, strict=True):
        pe = pe if isinstance(pe, dict) else {}
        for slot in ENTITY_SLOTS:
            gold, pred = ge.get(slot), pe.get(slot)
            if gold is not None:
                tp += pred == gold
                fn += pred != gold
            elif pred is not None:
                fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
