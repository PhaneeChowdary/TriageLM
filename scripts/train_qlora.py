"""QLoRA fine-tune with Unsloth. GPU-only, config-driven.

Needs CUDA, so it does not run on macOS. Run inside a notebook after:
    pip install -r requirements-gpu.txt
The trl and unsloth APIs drift between versions. Pin from `pip freeze` after
the first successful run, then commit that into requirements-gpu.txt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ticket_triage.config import load_config


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # Import unsloth first so it can patch trl/transformers/peft.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    m, lora, tr, out = cfg["model"], cfg["lora"], cfg["train"], cfg["output"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"],
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    def to_text(rec: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(rec["messages"], tokenize=False)}

    train_ds = Dataset.from_list(_load_jsonl(cfg["data"]["train_file"])).map(to_text)
    val_ds = Dataset.from_list(_load_jsonl(cfg["data"]["val_file"])).map(to_text)

    # Only log to W&B if a key is present; otherwise disable so it never blocks on login.
    use_wandb = bool(cfg["tracking"]["wandb_project"]) and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg["tracking"]["wandb_project"])
    else:
        os.environ["WANDB_MODE"] = "disabled"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=m["max_seq_length"],
            packing=tr["packing"],
            per_device_train_batch_size=tr["per_device_train_batch_size"],
            gradient_accumulation_steps=tr["gradient_accumulation_steps"],
            warmup_ratio=tr["warmup_ratio"],
            num_train_epochs=tr["num_train_epochs"],
            learning_rate=tr["learning_rate"],
            lr_scheduler_type=tr["lr_scheduler_type"],
            weight_decay=tr["weight_decay"],
            optim=tr["optim"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=cfg["seed"],
            logging_steps=10,
            output_dir=out["adapter_dir"],
            report_to="wandb" if use_wandb else "none",
            run_name=cfg["tracking"]["run_name"],
        ),
    )
    trainer.train()

    Path(out["adapter_dir"]).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out["adapter_dir"])
    tokenizer.save_pretrained(out["adapter_dir"])

    # Export GGUF before pushing so an auth failure can't cost the local artifacts.
    if out.get("export_gguf"):
        model.save_pretrained_gguf(
            out["adapter_dir"], tokenizer, quantization_method=out["gguf_quant"]
        )
        print("exported GGUF for Ollama re-eval")

    if out["push_to_hub"]:
        repo = out["hub_repo_id"]
        assert repo and "CHANGE_ME" not in repo, "Set output.hub_repo_id before pushing."
        model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)
        print("pushed adapter ->", repo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_qlora.yaml")
    main(p.parse_args().config)
