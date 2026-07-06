.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test smoke data explore eval-baseline eval-finetuned compare demo bundle clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create env and install base + dev deps
	uv venv
	uv pip install -e ".[dev,tracking]"

lint:  ## Ruff lint
	uv run ruff check src tests scripts

format:  ## Ruff format + import sort
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:  ## mypy
	uv run mypy

test:  ## Run unit tests
	uv run pytest

smoke:  ## End-to-end pipeline on ~50 rows with a mocked generator
	uv run python scripts/smoke_test.py

data:  ## Download Bitext, derive fields, reformat, and split
	uv run python scripts/run_pipeline.py --config configs/data.yaml

explore:  ## Print/save label distributions and dataset stats
	uv run python -m ticket_triage.data.explore --config configs/data.yaml

eval-baseline:  ## Run the base model through the eval harness, the fine-tune gate
	uv run python scripts/run_baseline.py --config configs/baseline.yaml

eval-finetuned:  ## Eval the fine-tuned model triage-ft through the harness
	uv run python scripts/run_baseline.py --config configs/finetuned.yaml

compare:  ## Merge saved eval reports into one comparison table
	uv run python scripts/compare_reports.py

demo:  ## Interactive live demo: type a ticket, get the triage JSON
	uv run python scripts/demo.py

bundle:  ## Zip code + processed data for Kaggle upload
	rm -f kaggle_bundle.zip
	zip -r -q kaggle_bundle.zip src scripts configs pyproject.toml requirements-gpu.txt \
		Makefile README.md data/processed/train.jsonl data/processed/val.jsonl \
		data/processed/test.jsonl -x '*/__pycache__/*' '*.DS_Store'
	@echo "built kaggle_bundle.zip"

clean:  ## Remove derived data and caches
	rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
