#!/usr/bin/env python3
"""LoRA SFT for ProofWriter answer, CoT+answer, and full-graph targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import activations

from prepare_proofwriter_training_data import build_record
from proofwriter_prompts import maybe_set_gemma_chat_template

# transformers>=4.52 removed this symbol, while AutoAWQ imported by PEFT
# still expects it.
activations.PytorchGELUTanh = activations.GELUTanh

SFT_MODES = ("answer_sft", "cot_answer_sft", "graph_sft")
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mode", choices=SFT_MODES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    parser.add_argument("--save-model-dir", default=None)
    return parser.parse_args()


def load_training_records(path: Path, mode: str) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    records = []
    for row in rows:
        record = row if row.get("method") == mode and row.get("target") else build_record(row, mode)
        target = str(record.get("target", "")).strip()
        prompt = record.get("prompt")
        if not target or not isinstance(prompt, list):
            continue
        records.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": target}],
                "target": target,
            }
        )
    if not records:
        raise ValueError("No valid SFT records produced")
    return records


def model_alias(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name.lower().replace("-", "_").replace(".", "_")


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    name = f"{model_alias(args.model_path)}_proofwriter_{args.mode}_lora_seed{args.seed}"
    return (
        args.output_dir or f"outputs_{name}",
        args.logging_dir or f"runs_{name}",
        args.save_model_dir or name,
    )


def main() -> None:
    # Heavy ML dependencies are intentionally imported only for an actual run.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    args = parse_args()
    set_seed(args.seed)
    records = load_training_records(args.dataset_json, args.mode)
    dataset = Dataset.from_list(records).shuffle(seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    maybe_set_gemma_chat_template(tokenizer, args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=LORA_TARGET_MODULES,
            inference_mode=False,
        ),
    )
    output_dir, logging_dir, save_model_dir = resolve_paths(args)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        bf16=True,
        fp16=False,
        max_grad_norm=0.1,
        max_length=args.max_seq_length,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print(f"train_examples: {len(dataset)}")
    trainer.train()
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    main()
