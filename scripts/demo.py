"""Live demo: type a support ticket, get the triage JSON from the fine-tuned model.

    python scripts/demo.py                       # interactive
    python scripts/demo.py "cancel order 12345"  # one shot
"""

from __future__ import annotations

import argparse
import json

from ticket_triage.eval.parse import parse
from ticket_triage.models.ollama import OllamaGenerator


def triage(gen: OllamaGenerator, ticket: str) -> None:
    result = parse(gen.generate(ticket))
    print(f"\nticket: {ticket}")
    if result.obj is not None:
        print(json.dumps(result.obj, indent=2))
    else:
        print("(model did not return parseable JSON)")
    print(f"schema valid: {result.schema_ok}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticket", nargs="*", help="ticket text; omit for interactive mode")
    p.add_argument("--model", default="triage-ft")
    args = p.parse_args()

    gen = OllamaGenerator(args.model, {}, prompt_style="trained")

    if args.ticket:
        triage(gen, " ".join(args.ticket))
        return

    print("Type a support ticket, blank line to quit.")
    while True:
        try:
            ticket = input("\nticket> ").strip()
        except EOFError:
            break
        if not ticket:
            break
        triage(gen, ticket)


if __name__ == "__main__":
    main()
