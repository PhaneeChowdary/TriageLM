"""Render eval reports as a markdown comparison table."""

from __future__ import annotations

COLUMNS = "| model | n | JSON % | schema % | intent F1 | cat acc | entity F1 | latency ms |"
SEP = "|---|---|---|---|---|---|---|---|"


def markdown_table(reports: dict[str, dict]) -> str:
    rows = [COLUMNS, SEP]
    for name, r in reports.items():
        rows.append(
            f"| {name} | {r['n']} | {r['json_parse_rate'] * 100:.1f} | "
            f"{r['schema_valid_rate'] * 100:.1f} | {r['intent_macro_f1']:.3f} | "
            f"{r['category_accuracy']:.3f} | {r['entity']['f1']:.3f} | "
            f"{r['latency_ms_mean']:.2f} |"
        )
    return "\n".join(rows)
