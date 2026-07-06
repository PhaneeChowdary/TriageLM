from ticket_triage.data.reformat import extract_entities, to_record
from ticket_triage.schema import validate_output


def test_extract_maps_placeholders_to_slots():
    ents = extract_entities("cancel order {{Order Number}} for {{Person Name}}")
    assert ents["order_id"] == "{{Order Number}}"
    assert ents["person_name"] == "{{Person Name}}"
    assert ents["refund_amount"] is None


def test_currency_symbol_folds_into_refund_amount():
    ents = extract_entities("refund of {{Currency Symbol}}{{Refund Amount}}")
    assert ents["refund_amount"] is not None


def test_to_record_is_schema_valid():
    row = {"intent": "cancel_order", "category": "ORDER",
           "instruction": "cancel order {{Order Number}}"}
    rec = to_record(row, 0)
    validate_output(rec["target"])  # raises if invalid
    assert rec["target"]["priority"] == "high"
    assert rec["messages"][-1]["role"] == "assistant"
