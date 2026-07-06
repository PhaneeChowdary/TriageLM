"""Generator interface so the eval harness never depends on a concrete model.

Tests use MockGenerator without a GPU or network; real runs use Ollama.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, ticket: str) -> str:
        """Return the raw model completion for one ticket."""


class MockGenerator(Generator):
    """Replays gold JSON keyed by ticket text; proves the harness offline.

    corrupt_every>0 truncates every Nth output to exercise the parse-failure path.
    """

    def __init__(self, gold_by_ticket: dict[str, dict], *, corrupt_every: int = 0):
        self._gold = gold_by_ticket
        self._corrupt_every = corrupt_every
        self._n = 0

    def generate(self, ticket: str) -> str:
        self._n += 1
        payload = json.dumps(self._gold.get(ticket, {}))
        if self._corrupt_every and self._n % self._corrupt_every == 0:
            return payload[:-3]  # truncated -> invalid JSON
        return payload
