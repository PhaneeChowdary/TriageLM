# TriageLM

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Turn a raw customer support message into a strict JSON object with a QLoRA-fine-tuned Llama-3.2-3B.

This is a "form, not facts" task: the model learns a fixed output schema, not how to
answer tickets. A visual
overview is in `docs/overview.html`.

One message from the test set and the JSON the model must return:

```text
help me check in which cases can i ask for reimbursements
```

```json
{
  "intent": "check_refund_policy",
  "category": "REFUND",
  "entities": {"order_id": null, "account_type": null, "account_category": null, "person_name": null,
               "refund_amount": null, "delivery_city": null, "delivery_country": null, "invoice_id": null},
  "priority": "medium",
  "requires_human": false,
  "suggested_queue": "billing"
}
```

## Results

Base vs the fine-tune, same 500-row test sample, both run through the eval harness
via Ollama.

| Model | Schema valid % | Intent macro-F1 | Category acc | Entity F1 | Prompt tokens |
| ------- | :---: | :---: | :---: | :---: | :---: |
| Base | 100 | 0.585 | 0.734 | 0.726 | 583 |
| Fine-tuned | 100 | **1.000** | **1.000** | **1.000** | **81** |

Robustness on the OOD slice:

| Model | Schema valid % | Intent macro-F1 | Category acc | Entity F1 |
| ------- | :---: | :---: | :---: | :---: |
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

## Private training settings

The training script loads `.env` from the project folder. Shell and notebook environment variables take precedence over `.env`. YAML provides the defaults.

Copy `.env.example` to `.env` on a new machine. Set `TRIAGE_HUB_REPO_ID` to your Hugging Face `owner/repository`. Set `TRIAGE_PUSH_TO_HUB=true` to upload after training and local export. Use your existing Hugging Face login or provide `HF_TOKEN` privately. No token is included in the example.

For a local-only run, override the upload setting for that command:

```bash
TRIAGE_PUSH_TO_HUB=false python scripts/train_qlora.py --config configs/train_qlora.yaml
```

This switch only controls uploads. Training and local model files work with either setting. QLoRA training still requires a supported CUDA GPU; this change does not add Mac training support.

`.env` is ignored by Git and is not included by `make bundle`. On Kaggle, set the two `TRIAGE_` variables in the notebook environment before running the training script. Keep the Hugging Face token in Kaggle Secrets. With no private settings, bundled code saves locally without uploading.
