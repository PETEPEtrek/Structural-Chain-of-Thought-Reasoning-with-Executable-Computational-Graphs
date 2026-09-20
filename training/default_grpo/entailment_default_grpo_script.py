#!/usr/bin/env python3
"""
Default GRPO training for EntailmentBank baselines on original data.
Uses format rewards + native proof-string matching (no numeric reward).
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    activations,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer

from entailment_baseline_prompts import (
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    SYSTEM_PROMPT_DEFAULT_GRPO,
    build_proof_target,
    build_user_input,
    normalize_proof,
)

activations.PytorchGELUTanh = activations.GELUTanh

MODEL_PATHS = {
    "llama3_2_3b": "models/petepetrek/meta-llama/Llama-3.2-3B-Instruct",
    "qwen3_4b": "models/petepetrek/Qwen/Qwen3-4B-Instruct-2507",
    "falcon3_3b": "models/petepetrek/tiiuae/Falcon3-3B-Instruct",
    "gemma2_2b": "models/petepetrek/google/gemma-2-2b-it",
}

MODEL_ALIASES = {
    "llama3_2_3b": "llama3_2",
    "qwen3_4b": "qwen3_4b",
    "falcon3_3b": "falcon3",
    "gemma2_2b": "gemma_2_2b",
}

GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"""

MAX_SEQ_LENGTH = 2048
LORA_RANK = 32

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["llama3_2_3b", "qwen3_4b", "falcon3_3b", "gemma2_2b"],
        required=True,
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Baseline JSON from prepare_entailmentbank_baseline_data.py.",
    )
    parser.add_argument("--task-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    parser.add_argument("--save-model-dir", default=None)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def is_gemma_2_2b(path_like: str) -> bool:
    value = path_like.lower()
    return "gemma-2-2b" in value or "gemma_2_2b" in value


def maybe_set_gemma_chat_template(tokenizer, model_path: str) -> None:
    if is_gemma_2_2b(model_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 3.0 if MATCH_FORMAT.search(response) is not None else 0.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        score += 0.5 if response.count(REASONING_START) == 1 else -1.0
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def check_proof(prompts, completions, proof, **kwargs):
    global PRINTED_TIMES
    responses = [c[0]["content"] for c in completions]
    extracted = []
    for response in responses:
        match = MATCH_FORMAT.search(response)
        extracted.append(match.group(1).strip() if match else None)

    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print("⎯" * 20)
        print("Prompt:\n", prompts[0][-1]["content"])
        print("True Proof:\n", proof[0])
        print("Response:\n", responses[0])
        print("Extracted Proof:\n", extracted[0])
        print("⎯" * 20)
    PRINTED_TIMES += 1

    scores = []
    for guess, true in zip(extracted, proof):
        if guess is None:
            scores.append(0.0)
            continue
        norm_guess = normalize_proof(guess)
        norm_true = normalize_proof(true)
        if norm_guess == norm_true:
            scores.append(3.0)
        elif norm_true in norm_guess or norm_guess in norm_true:
            scores.append(1.5)
        else:
            scores.append(-1.0)
    return scores


def build_training_dataset(dataset_json: Path) -> Dataset:
    rows = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {dataset_json}")

    dataset_data = []
    for item in rows:
        user_input = build_user_input(item)
        proof = build_proof_target(item)
        if not user_input or not proof:
            continue
        dataset_data.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_DEFAULT_GRPO},
                    {"role": "user", "content": user_input},
                ],
                "proof": proof,
            }
        )
    if not dataset_data:
        raise ValueError(f"Training dataset is empty: {dataset_json}")
    return Dataset.from_list(dataset_data)


def resolve_paths(args, model_alias: str):
    save_name = (
        f"{model_alias}_entailment_task{args.task_id}_default_grpo_"
        f"{args.max_steps}_{args.num_generations}_lora_seed{args.seed}"
    )
    output_dir = args.output_dir or f"outputs_{save_name}"
    logging_dir = args.logging_dir or f"runs_{save_name}"
    save_model_dir = args.save_model_dir or save_name
    return output_dir, logging_dir, save_model_dir, save_name


def main():
    args = parse_args()
    set_global_seed(args.seed)

    model_path = MODEL_PATHS[args.model]
    model_alias = MODEL_ALIASES[args.model]
    output_dir, logging_dir, save_model_dir, save_name = resolve_paths(args, model_alias)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    maybe_set_gemma_chat_template(tokenizer, model_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
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
    model = get_peft_model(model, lora_config)

    dataset = build_training_dataset(args.dataset_json).shuffle(seed=args.seed)
    print(f"  train_examples: {len(dataset)}")

    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="tensorboard",
        logging_dir=logging_dir,
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_proof,
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    print("Resolved config:")
    print(f"  model_path: {model_path}")
    print(f"  dataset_json: {args.dataset_json}")
    print(f"  task_id: {args.task_id}")
    print(f"  save_model_dir: {save_model_dir}")

    trainer.train()
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    main()
