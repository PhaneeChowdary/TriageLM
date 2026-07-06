"""Corruption transforms for the out-of-distribution eval slice.

Inputs are perturbed but gold targets are unchanged. Deterministic given a seed.
"""

from __future__ import annotations

import random
import string
from copy import deepcopy

PREAMBLES = [
    "Hi there, hope you're having a great day!",
    "Quick one for you -",
    "Sorry to bother you, but",
    "btw I really like your service.",
]


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    out = list(text)
    for i, ch in enumerate(out):
        if ch.isalpha() and rng.random() < rate:
            out[i] = rng.choice(string.ascii_lowercase)
    return "".join(out)


def add_preamble(text: str, rng: random.Random) -> str:
    return f"{rng.choice(PREAMBLES)} {text}"


def concat_tickets(text: str, other: str) -> str:
    return f"{text} Also, {other}"


def corrupt_records(records: list[dict], cfg: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    t = cfg["transforms"]
    out = []
    for i, rec in enumerate(records):
        rec = deepcopy(rec)
        text = rec["instruction"]
        if t["typos"]["enabled"]:
            text = add_typos(text, t["typos"]["rate"], rng)
        if t["irrelevant_preamble"]["enabled"]:
            text = add_preamble(text, rng)
        if t["concat_tickets"]["enabled"]:
            text = concat_tickets(text, records[(i + 1) % len(records)]["instruction"])
        rec["instruction"] = text
        out.append(rec)
    return out
