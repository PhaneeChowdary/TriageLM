from ticket_triage.eval.metrics import category_accuracy, entity_prf, intent_macro_f1


def test_perfect_intent_and_category():
    g = ["a", "b", "a"]
    assert intent_macro_f1(g, g) == 1.0
    assert category_accuracy(g, g) == 1.0


def test_entity_prf_counts_tp_fp_fn():
    gold = [{"order_id": "1", "refund_amount": None}]
    pred = [{"order_id": "1", "refund_amount": "9"}]  # tp order_id, fp refund_amount
    r = entity_prf(pred, gold)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_entity_missed_gold_is_fn():
    gold = [{"order_id": "1"}]
    pred = [{"order_id": None}]
    r = entity_prf(pred, gold)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)


def test_entity_prf_handles_non_dict_pred():
    r = entity_prf([None], [{"order_id": "1"}])
    assert r["fn"] == 1
