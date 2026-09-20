#!/usr/bin/env python3
"""
SFT training for EntailmentBank baselines on original data:
- answer_sft: compact native proof string
- cot_plus_answer_sft: full_text_proof + ANSWER: proof
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, activations
from trl import SFTConfig, SFTTrainer

from entailment_baseline_prompts import (
    SYSTEM_PROMPT_ANSWER_SFT,
    SYSTEM_PROMPT_COT_ANSWER_SFT,
    build_answer_target,
    build_cot_plus_answer_target,
    build_proof_target,
    build_user_input,
)

activations.PytorchGELUTanh = activations.GELUTanh

MODEL_ALIASES = {
    "llama3_2_3b": "llama3_2",
    "qwen3_4b": "qwen3_4b",
    "falcon3_3b": "falcon3",
    "gemma2_2b": "gemma_2_2b",
}

GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Baseline JSON from prepare_entailmentbank_baseline_data.py.",
    )
    parser.add_argument(
        "--config",
        choices=("answer_sft", "cot_plus_answer_sft"),
        required=True,
    )
    parser.add_argument("--task-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    parser.add_argument("--save-model-dir", default=None)
    return parser.parse_args()


def model_alias(model_path: str) -> str:
    low = model_path.lower()
    if "llama-3.2-3b" in low:
        return MODEL_ALIASES["llama3_2_3b"]
    if "qwen3-4b" in low:
        return MODEL_ALIASES["qwen3_4b"]
    if "falcon3-3b" in low:
        return MODEL_ALIASES["falcon3_3b"]
    if "gemma-2-2b" in low:
        return MODEL_ALIASES["gemma2_2b"]
    return "model"


def is_gemma_2_2b(path_like: str) -> bool:
    path = path_like.lower()
    return "gemma-2-2b" in path or "gemma_2_2b" in path


def maybe_set_gemma_chat_template(tokenizer, model_path: str) -> None:
    if is_gemma_2_2b(model_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM


def build_dataset(dataset_json: Path, config: str) -> Dataset:
    rows = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {dataset_json}")

    target_fn = (
        build_answer_target
        if config == "answer_sft"
        else build_cot_plus_answer_target
    )
    system_prompt = (
        SYSTEM_PROMPT_ANSWER_SFT
        if config == "answer_sft"
        else SYSTEM_PROMPT_COT_ANSWER_SFT
    )
    data = []
    for item in rows:
        user_input = build_user_input(item)
        completion = target_fn(item)
        if not user_input.strip() or not completion.strip() or not build_proof_target(item):
            continue
        data.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    if not data:
        raise ValueError("No training examples produced.")
    return Dataset.from_list(data)


def resolve_paths(args: argparse.Namespace):
    alias = model_alias(args.model_path)
    mode_tag = "answer" if args.config == "answer_sft" else "cot_plus_answer"
    base = f"{alias}_sft_entailment_task{args.task_id}_{mode_tag}_lora/seed{args.seed}"
    output_dir = args.output_dir or f"outputs_{base}"
    logging_dir = args.logging_dir or f"runs_{base}"
    save_model_dir = args.save_model_dir or base
    return output_dir, logging_dir, save_model_dir


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.dataset_json, args.config).shuffle(seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    maybe_set_gemma_chat_template(tokenizer, args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)

    output_dir, logging_dir, save_model_dir = resolve_paths(args)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=250,
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        bf16=True,
        fp16=False,
        max_grad_norm=0.1,
        save_total_limit=1,
        max_length=args.max_seq_length,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print(f"  train_examples: {len(dataset)}")
    print(f"  save_model_dir: {save_model_dir}")
    trainer.train()
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    main()
