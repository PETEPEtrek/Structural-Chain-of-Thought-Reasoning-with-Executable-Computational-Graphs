#!/usr/bin/env python3
"""
Default GRPO training script (non-graph baseline) based on advanced_grpo2.py.
"""

import argparse
import random
import re
from fractions import Fraction

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

DATASET_FILE_MAP = {
    "gsm8k": "gsm8k_comp_graphs_train.json",
    "igsm": "igsm_openrouter_valid_train.json",
    "math": "math_openrouter_valid_train.json",
    "gsm8k_new_ops": "gsm8k_new_ops_comp_graphs_train.json",
}

MAX_SEQ_LENGTH = 1024
LORA_RANK = 32

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

MATCH_NUMBERS = re.compile(
    SOLUTION_START + r".*?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    flags=re.MULTILINE | re.DOTALL,
)

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Default GRPO training script.")
    parser.add_argument(
        "--model",
        choices=["llama3_2_3b", "qwen3_4b", "falcon3_3b", "gemma2_2b"],
        default="qwen3_4b",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "gsm8k",
            "igsm",
            "math",
            "gsm8k_new_ops",
        ],
        default="gsm8k",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Trainer token cap for conditioned prompt (questions + templates).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Trainer token cap for generated completions.",
    )
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


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def get_question_text(item: dict, dataset_name: str) -> str:
    if dataset_name == "igsm":
        # OpenRouter IGSM files use "question"; keep backward compatibility
        # with older variants that used "problem".
        return item.get("question") or item.get("problem", "")
    return item.get("question", "")


def get_target_answer(item: dict, dataset_name: str):
    if dataset_name in {"igsm", "math", "gsm8k_new_ops"}:
        answer = item.get("answer")
        return None if answer is None else str(answer).strip()
    return extract_hash_answer(item.get("solution", ""))


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


def check_answer(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = []
    for response in responses:
        match = MATCH_FORMAT.search(response)
        extracted.append(match.group(1) if match else None)
    scores = []
    for guess, true in zip(extracted, answer):
        score = 0.0
        if guess is None:
            scores.append(0.0)
            continue
        if guess == true:
            score += 3.0
        elif guess.strip() == true.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true)
                if 0.9 <= ratio <= 1.1:
                    score += 1.0
                elif 0.8 <= ratio <= 1.2:
                    score += 0.5
                else:
                    score -= 1.5
            except Exception:
                score -= 1.5
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    global PRINTED_TIMES
    question = prompts[0][-1]["content"]
    responses = [c[0]["content"] for c in completions]
    extracted = []
    for response in responses:
        match = MATCH_NUMBERS.search(response)
        extracted.append(match.group(1) if match else None)

    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print("⎯" * 20)
        print("Question:\n", question)
        print("True Answer:\n", answer[0])
        print("Response:\n", responses[0])
        print("Extracted Number:\n", extracted[0])
        print("⎯" * 20)
    PRINTED_TIMES += 1

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            def parse_numeric(value: str) -> float:
                clean = value.strip().replace(",", "")
                if "/" in clean:
                    return float(Fraction(clean))
                return float(clean)

            true_val = parse_numeric(true)
            guess_val = parse_numeric(guess)
            scores.append(1.5 if guess_val == true_val else -0.5)
        except Exception:
            scores.append(0.0)
    return scores


def build_training_dataset(json_file: str, dataset_name: str) -> Dataset:
    import json

    with open(json_file, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    dataset_data = []
    for item in data:
        answer = get_target_answer(item, dataset_name)
        question = get_question_text(item, dataset_name)
        if not question or answer is None:
            continue
        dataset_data.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
            }
        )
    if not dataset_data:
        raise ValueError(
            "Training dataset is empty after preprocessing. "
            f"dataset_name={dataset_name}, json_file={json_file}. "
            "Check question/answer field mapping."
        )
    return Dataset.from_list(dataset_data)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    model_path = MODEL_PATHS[args.model]
    model_alias = MODEL_ALIASES[args.model]
    dataset_file = DATASET_FILE_MAP[args.dataset]

    save_name = (
        f"{model_alias}_{args.dataset}_default_grpo_default_scheduler_"
        f"{args.max_steps}_no_rewards_{args.num_generations}_lora_seed{args.seed}"
    )
    output_dir = f"outputs_{save_name}"
    logging_dir = f"runs_{save_name}"

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

    dataset = build_training_dataset(json_file=dataset_file, dataset_name=args.dataset)
    dataset = dataset.shuffle(seed=args.seed)
    print(f"  train_examples: {len(dataset)}")

    #max_prompt_length = 
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
            check_answer,
            check_numbers,
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    print("Resolved config:")
    print(f"  model_path: {model_path}")
    print(f"  dataset_file: {dataset_file}")
    print(f"  max_prompt_length: {args.max_prompt_length}")
    print(f"  max_completion_length: {args.max_completion_length}")
    print(f"  output_dir: {output_dir}")
    print(f"  logging_dir: {logging_dir}")
    print(f"  save_name: {save_name}")

    trainer.train()
    trainer.save_model(save_name)
    tokenizer.save_pretrained(save_name)


if __name__ == "__main__":
    main()
