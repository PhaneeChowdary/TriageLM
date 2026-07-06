"""Ollama-backed generator for local baseline inference."""

from __future__ import annotations

import json
import sys
import urllib.request

from ticket_triage.models.generator import Generator
from ticket_triage.models.prompts import build_baseline_messages, build_messages


class OllamaGenerator(Generator):
    def __init__(
        self,
        model: str,
        vocab: dict,
        *,
        shots: list[tuple[str, dict]] = (),
        prompt_style: str = "schema",  # "schema" for the base model, "trained" for the fine-tune
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
    ):
        self.model = model
        self.vocab = vocab
        self.shots = list(shots)
        self.prompt_style = prompt_style
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _messages(self, ticket: str) -> list[dict]:
        if self.prompt_style == "trained":
            return build_messages(ticket)  # short training prompt, no few-shot
        return build_baseline_messages(ticket, self.vocab, self.shots)

    def generate(self, ticket: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": self._messages(ticket),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except Exception as e:  # one bad call shouldn't kill a long run
            print(f"ollama request failed: {e}", file=sys.stderr)
            return ""
