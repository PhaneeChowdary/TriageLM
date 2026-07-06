from ticket_triage.data.split import stratified_split

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _rows(n_per_intent=20, intents=("a", "b", "c")):
    return [{"target": {"intent": i}} for i in intents for _ in range(n_per_intent)]


def test_splits_partition_all_rows_without_overlap():
    rows = _rows()
    s = stratified_split(rows, RATIOS, "intent", seed=42)
    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert total == len(rows)
    ids = [id(r) for part in s.values() for r in part]
    assert len(ids) == len(set(ids))


def test_deterministic_under_same_seed():
    rows = _rows()
    a = stratified_split(rows, RATIOS, "intent", seed=42)
    b = stratified_split(rows, RATIOS, "intent", seed=42)
    assert [r["target"] for r in a["test"]] == [r["target"] for r in b["test"]]
