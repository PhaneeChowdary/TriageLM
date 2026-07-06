"""LLM-as-judge scaffold. Off by default; not wired to a backend yet."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Judge(ABC):
    @abstractmethod
    def score(self, instruction: str, prediction: dict) -> dict:
        """Return a judgment, e.g. {'valid': bool, 'reason': str}."""


class NullJudge(Judge):
    def score(self, instruction: str, prediction: dict) -> dict:
        return {"enabled": False}
