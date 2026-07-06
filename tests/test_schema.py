"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from ticket_triage.schema import Priority, TriageOutput, validate_output

VALID = {
    "intent": "cancel_order",
    "category": "ORDER",
    "entities": {"order_id": "12345", "refund_amount": None},
    "priority": "high",
    "requires_human": False,
    "suggested_queue": "orders-l1",
}


def test_valid_object_parses():
    out = validate_output(VALID)
    assert isinstance(out, TriageOutput)
    assert out.priority is Priority.high
    assert out.entities.order_id == "12345"


def test_entities_default_to_none():
    payload = {**VALID}
    del payload["entities"]
    out = validate_output(payload)
    assert out.entities.order_id is None


def test_extra_top_level_field_rejected():
    with pytest.raises(ValidationError):
        validate_output({**VALID, "unexpected": "x"})


def test_extra_entity_slot_rejected():
    bad = {**VALID, "entities": {"order_id": "1", "sku": "abc"}}
    with pytest.raises(ValidationError):
        validate_output(bad)


def test_bad_priority_rejected():
    with pytest.raises(ValidationError):
        validate_output({**VALID, "priority": "urgent"})
