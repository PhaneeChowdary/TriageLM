"""The derivation rules must cover every gold label in configs/schema.yaml."""

from pathlib import Path

import yaml

from ticket_triage.data.derive import derive, priority, requires_human, suggested_queue

VOCAB = yaml.safe_load(Path("configs/schema.yaml").read_text())


def test_priority_valid_for_every_intent():
    for intent in VOCAB["intents"]:
        assert priority(intent) in VOCAB["priorities"]


def test_every_category_maps_to_a_known_queue():
    for category in VOCAB["categories"]:
        assert suggested_queue(category) in VOCAB["queues"]


def test_escalated_intents_require_human():
    assert requires_human("complaint")
    assert requires_human("payment_issue")
    assert not requires_human("place_order")


def test_derive_shape():
    out = derive("payment_issue", "PAYMENT")
    assert out == {"priority": "high", "requires_human": True, "suggested_queue": "billing"}
