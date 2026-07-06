"""Parse a raw model completion into a schema-validated object."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ticket_triage.schema import validate_output


@dataclass
class ParseResult:
    obj: dict | None
    json_ok: bool
    schema_ok: bool


def _load_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse(text: str) -> ParseResult:
    obj = _load_json(text)
    if obj is None:
        return ParseResult(None, json_ok=False, schema_ok=False)
    try:
        validate_output(obj)
    except Exception:
        return ParseResult(obj, json_ok=True, schema_ok=False)
    return ParseResult(obj, json_ok=True, schema_ok=True)
