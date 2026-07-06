"""Chat prompt construction for triage."""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a support-ticket triage system. Read the customer message and return "
    "ONLY a JSON object with keys: intent, category, entities, priority, "
    "requires_human, suggested_queue. No prose."
)


def build_messages(instruction: str, target: dict | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": json.dumps(target)})
    return msgs


def schema_system_prompt(vocab: dict) -> str:
    """Long, schema-laden prompt for the base-model baseline."""
    return (
        "You are a support-ticket triage system. Convert the customer message into a "
        "JSON object with EXACTLY these keys: intent, category, entities, priority, "
        "requires_human, suggested_queue.\n"
        f"intent: one of {', '.join(vocab['intents'])}.\n"
        f"category: one of {', '.join(vocab['categories'])}.\n"
        f"entities: object with keys {', '.join(vocab['entity_slots'])}; null when absent.\n"
        f"priority: one of {', '.join(vocab['priorities'])}.\n"
        "requires_human: boolean.\n"
        f"suggested_queue: one of {', '.join(vocab['queues'])}.\n"
        "Return ONLY the JSON object, no prose."
    )


def build_baseline_messages(
    instruction: str, vocab: dict, shots: list[tuple[str, dict]] = ()
) -> list[dict]:
    msgs = [{"role": "system", "content": schema_system_prompt(vocab)}]
    for shot_instruction, shot_target in shots:
        msgs.append({"role": "user", "content": shot_instruction})
        msgs.append({"role": "assistant", "content": json.dumps(shot_target)})
    msgs.append({"role": "user", "content": instruction})
    return msgs
