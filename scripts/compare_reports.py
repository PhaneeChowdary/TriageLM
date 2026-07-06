"""Merge saved eval reports into one comparison table."""

from __future__ import annotations

import glob
import json
from pathlib import Path

from ticket_triage.eval.report import markdown_table


def main() -> None:
    reports: dict[str, dict] = {}
    for path in sorted(glob.glob("reports/baseline_*.json")):
        reports.update(json.loads(Path(path).read_text()))
    print(markdown_table(reports))


if __name__ == "__main__":
    main()
