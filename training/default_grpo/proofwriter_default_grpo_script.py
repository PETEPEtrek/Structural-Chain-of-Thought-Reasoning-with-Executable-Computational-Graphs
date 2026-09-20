#!/usr/bin/env python3
"""Default ProofWriter GRPO: format and boolean-answer rewards only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import activations

from prepare_proofwriter_training_data import build_record
from proofwriter_prompts import (
    FULL_RESPONSE_RE,
    SOLUTION_END,
    SOLUTION_START,
    THINK_END,
    THINK_START,
    maybe_set_gemma_chat_template,
    normalize_boolean,
)

# transformers>=4.52 removed this symbol, while AutoAWQ imported by PEFT
# still expects it.
activations.PytorchGELUTanh = activations.GELUTanh

DEFAULT_NUM_GENERATIONS = 16
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def _content(completion: Any) -> str:
    try:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        return str(completion[0].get("content", ""))
    except (AttributeError, IndexError, TypeError):
        return ""


def match_format_exactly(completions, **kwargs) -> list[float]:
    """GSM8K-style +3 reward; regex anchors the full response."""
    return [3.0 if FULL_RESPONSE_RE.fullmatch(_content(c)) else 0.0 for c in completions]


def match_format_approximately(completions, **kwargs) -> list[float]:
    """GSM8K-style tag-count shaping: +0.5 once, otherwise -1 per tag."""
    scores = []
    for completion in completions:
        response = _content(completion)
        score = 0.0
        for tag in (THINK_START, THINK_END, SOLUTION_START, SOLUTION_END):
            score += 0.5 if response.count(tag) == 1 else -1.0
        scores.append(score)
    return scores


def check_boolean_answer(completions, answer, **kwargs) -> list[float]:
    """Reward the normalized boolean contained inside a valid solution element."""
    scores = []
    for completion, gold in zip(completions, answer):
        match = FULL_RESPONSE_RE.fullmatch(_content(completion))
        if not match:
            scores.append(0.0)
            continue
        try:
            correct = match.group("answer").casefold() == normalize_boolean(gold).casefold()
        except ValueError:
            correct = False
        scores.append(3.0 if correct else -1.5)
    return scores


_PLACEHOLDER_THINK = {
    "nonempty reasoning",
    "your reasoning here",
    "...step-by-step reasoning...",
    "step-by-step reasoning",
}


def check_think_is_real(completions, **kwargs) -> list[float]:
    """Penalize empty/placeholder <think> bodies; lightly reward rule-grounded CoT."""
    scores = []
    for completion in completions:
        match = FULL_RESPONSE_RE.fullmatch(_content(completion))
        if not match:
            scores.append(0.0)
            continue
        think = match.group("think").strip().casefold()
        if not think or think in _PLACEHOLDER_THINK:
            scores.append(-1.5)
            continue
        if "apply rule" in think or "given " in think:
            scores.append(1.0)
        else:
            scores.append(0.5)
    return scores


def load_training_records(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    records = []
    for row in rows:
        record = (
            row
            if row.get("method") == "default_grpo" and row.get("prompt")
            else build_record(row, "default_grpo")
        )
        records.append({"prompt": record["prompt"], "answer": record["answer"]})
    if not records:
        raise ValueError("No default GRPO records produced")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Optional CoT/answer SFT LoRA to continue from (recommended: cot_answer_sft).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-model-dir", default=None)
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    records = load_training_records(args.dataset_json)
    dataset = Dataset.from_list(records).shuffle(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    maybe_set_gemma_chat_template(tokenizer, args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if args.sft_checkpoint:
        model = PeftModel.from_pretrained(
            base, args.sft_checkpoint, is_trainable=True
        )
        print(f"loaded SFT adapter: {args.sft_checkpoint}")
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=LORA_TARGET_MODULES,
                inference_mode=False,
            ),
        )
    name = f"{Path(args.model_path).name}_proofwriter_default_grpo_{args.num_generations}_seed{args.seed}"
    output_dir = args.output_dir or f"outputs_{name}"
    config = GRPOConfig(
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
        save_strategy="no",
        max_grad_norm=0.1,
        report_to="tensorboard",
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        fp16=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_boolean_answer,
            check_think_is_real,
        ],
        args=config,
        train_dataset=dataset,
    )
    print(f"train_examples: {len(dataset)}")
    trainer.train()
    save_model_dir = args.save_model_dir or name
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    main()
