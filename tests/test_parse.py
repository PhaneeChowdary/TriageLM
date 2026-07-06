from ticket_triage.eval.parse import parse

VALID = {
    "intent": "cancel_order",
    "category": "ORDER",
    "entities": {"order_id": "{{Order Number}}"},
    "priority": "high",
    "requires_human": False,
    "suggested_queue": "orders",
}


def _json(obj):
    import json

    return json.dumps(obj)


def test_valid_output_parses_and_validates():
    r = parse(_json(VALID))
    assert r.json_ok and r.schema_ok


def test_json_wrapped_in_prose_is_recovered():
    r = parse("Sure! Here you go:\n" + _json(VALID) + "\nHope that helps.")
    assert r.json_ok and r.schema_ok


def test_broken_json_fails_both():
    r = parse("{intent: cancel_order,,,")
    assert not r.json_ok and not r.schema_ok


def test_valid_json_but_bad_schema():
    r = parse(_json({**VALID, "priority": "urgent"}))
    assert r.json_ok and not r.schema_ok
