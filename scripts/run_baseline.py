"""Run the base model through the eval harness via Ollama. The fine-tune gate."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from ticket_triage.config import load_config
from ticket_triage.eval.harness import evaluate
from ticket_triage.eval.report import markdown_table
from ticket_triage.eval.robustness import corrupt_records
from ticket_triage.models.ollama import OllamaGenerator
from ticket_triage.utils.seed import set_seed


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _check_ollama(host: str) -> None:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=5).read()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {host} ({e}). Run `ollama serve`.") from e


def main(config_path: str, limit: int | None) -> None:
    cfg = load_config(config_path)
    ecfg = load_config(cfg["eval_config"])
    dcfg = load_config("configs/data.yaml")
    vocab = load_config("configs/schema.yaml")
    set_seed(cfg["seed"])

    host = "http://localhost:11434"
    _check_ollama(host)

    processed = Path(dcfg["paths"]["processed_dir"])
    test = _load_jsonl(processed / "test.jsonl", limit)
    strategy = cfg["prompt"]["strategy"]
    shots = (
        [(r["instruction"], r["target"]) for r in
         _load_jsonl(processed / "train.jsonl", cfg["prompt"]["num_shots"])]
        if strategy == "few_shot"
        else []
    )
    prompt_style = "trained" if strategy == "trained" else "schema"

    gen = OllamaGenerator(
        cfg["model"]["name"], vocab, shots=shots, prompt_style=prompt_style, host=host,
        temperature=cfg["decoding"]["temperature"],
        max_tokens=cfg["decoding"]["max_new_tokens"],
    )

    ood = corrupt_records(test, ecfg["robustness"], ecfg["robustness"]["seed"])
    reports = {
        f"{cfg['model']['name']} (clean)": evaluate(gen, test),
        f"{cfg['model']['name']} (OOD)": evaluate(gen, ood),
    }
    table = markdown_table(reports)
    print(table)

    out = Path(ecfg["reporting"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    slug = cfg["model"]["name"].replace(":", "_").replace("/", "_")
    (out / f"baseline_{slug}.md").write_text(table + "\n")
    (out / f"baseline_{slug}.json").write_text(json.dumps(reports, indent=2))
    print(f"\nSaved -> {out}/baseline_{slug}.(md|json)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N test rows")
    args = p.parse_args()
    main(args.config, args.limit)
