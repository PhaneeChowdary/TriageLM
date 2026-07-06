"""CI-safe smoke: build synthetic records, run the harness with a mock."""

from ticket_triage.data.reformat import to_record
from ticket_triage.eval.harness import evaluate
from ticket_triage.eval.robustness import corrupt_records
from ticket_triage.models.generator import MockGenerator

ROWS = [
    {"intent": "cancel_order", "category": "ORDER", "instruction": "cancel order {{Order Number}}"},
    {"intent": "get_refund", "category": "REFUND", "instruction": "refund {{Refund Amount}}"},
    {"intent": "create_account", "category": "ACCOUNT", "instruction": "how do I sign up"},
    {"intent": "track_order", "category": "ORDER", "instruction": "where is my order"},
]
RECORDS = [to_record(r, i) for i, r in enumerate(ROWS)]

ROBUSTNESS = {
    "transforms": {
        "typos": {"enabled": True, "rate": 0.1},
        "concat_tickets": {"enabled": True, "n": 2},
        "irrelevant_preamble": {"enabled": True},
    }
}


def _mock(records, **kw):
    return MockGenerator({r["instruction"]: r["target"] for r in records}, **kw)


def test_perfect_mock_scores_perfectly():
    rep = evaluate(_mock(RECORDS), RECORDS)
    assert rep["json_parse_rate"] == 1.0
    assert rep["schema_valid_rate"] == 1.0
    assert rep["intent_macro_f1"] == 1.0
    assert rep["category_accuracy"] == 1.0
    assert rep["entity"]["f1"] == 1.0


def test_corruption_drops_schema_valid_rate():
    rep = evaluate(_mock(RECORDS, corrupt_every=2), RECORDS)
    assert rep["schema_valid_rate"] < 1.0


def test_ood_slice_same_size_and_runs():
    ood = corrupt_records(RECORDS, ROBUSTNESS, seed=1234)
    assert len(ood) == len(RECORDS)
    rep = evaluate(_mock(ood), ood)
    assert rep["n"] == len(RECORDS)
