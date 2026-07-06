# TriageLM

Turn a raw customer support message into a strict JSON object (intent, category,
entities, priority, routing) with a QLoRA-fine-tuned Llama-3.2-3B.

This is a "form, not facts" task: the model learns a fixed output schema, not how to
answer tickets (answering is a retrieval problem, and is out of scope). A visual
overview is in `docs/overview.html`.

## Results

Base vs the fine-tune, same 500-row test sample, both run through the eval harness
via Ollama.

| Model | Schema valid % | Intent macro-F1 | Category acc | Entity F1 | Prompt tokens |
|-------|:---:|:---:|:---:|:---:|:---:|
| Base (schema + few-shot prompt) | 100 | 0.585 | 0.734 | 0.726 | 583 |
| Fine-tuned (short prompt) | 100 | **1.000** | **1.000** | **1.000** | **81** |

Robustness on the OOD slice (typos, concatenated tickets, irrelevant preamble):

| Model | Schema valid % | Intent macro-F1 | Category acc | Entity F1 |
|-------|:---:|:---:|:---:|:---:|
| Base | 84.0 | 0.344 | 0.376 | 0.393 |
| Fine-tuned | **99.8** | **0.876** | **0.914** | **0.634** |

The clean 1.000 is inflated: Bitext is synthetic and templated, so the test set shares
templates with training. The OOD numbers are the honest signal.

## When you should NOT fine-tune this

- You need the model to answer or resolve tickets. That is RAG, not fine-tuning.
- The schema or label set changes often. Every change means re-training.
- A prompt-only baseline already clears your bar. Measure first.
- You cannot measure it. Build the eval harness before the model.

## How to run

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
and a Kaggle GPU for the training step.

```bash
# 1. Local pipeline + tests (no GPU)
make setup && make test && make data

# 2. Baseline gate (Ollama)
ollama pull llama3.2:3b
make eval-baseline

# 3. Fine-tune on Kaggle
make bundle        # upload kaggle_bundle.zip, then run notebooks/train_qlora.ipynb

# 4. Evaluate the fine-tune and compare
ollama create triage-ft -f Modelfile
make eval-finetuned
make compare

# Live demo: type a ticket, get the JSON
make demo
```

## Repo layout

```
configs/             YAML run configs
src/ticket_triage/   schema, data pipeline, eval harness, model interfaces
scripts/             pipeline, eval, training, demo
tests/               unit tests
notebooks/           train_qlora.ipynb (QLoRA training on Kaggle)
```

## License

MIT, see [LICENSE](LICENSE). The Bitext dataset is Apache-2.0 and the base model
Llama-3.2 is under the Llama Community License.
