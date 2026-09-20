#!/usr/bin/env python3
"""
vLLM inference for default GRPO LoRA checkpoints.

Designed for checkpoints that output reasoning + final answer, e.g. with:
<SOLUTION> ... </SOLUTION>
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

from tqdm import tqdm

DEFAULT_GRPO_SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    "Place it between <start_working_out> and <end_working_out>.\n"
    "Then, provide your solution between <SOLUTION></SOLUTION>"
)


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def extract_numeric_from_text(text: str, force_solution_answer: bool = True):
    if force_solution_answer:
        match = re.search(
            r"<SOLUTION>(.*?)</SOLUTION>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            solution_block = match.group(1).replace(",", "")
            numbers = re.findall(r"-?\d+(?:\.\d+)?", solution_block)
            if not numbers:
                return None
            try:
                return float(numbers[-1])
            except Exception:
                return None
        return None

    if "####" in text:
        maybe = text.split("####", 1)[1].strip()
        try:
            return float(maybe.replace(",", ""))
        except Exception:
            pass

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def to_json_safe_real(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, complex):
        if value.imag != 0.0:
            return None
        value = value.real
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, complex):
        return None
    return to_json_safe_real(value)


def load_split(path, split):
    """Read either a flat list of records or a {"train": [...], "test": [...]} bundle."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if split not in data:
            raise KeyError(f"Split '{split}' not in {path}; available: {sorted(data)}")
        return data[split]
    return data


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        fallback_messages = [
            {
                "role": "user",
                "content": (
                    f"System instruction:\n{DEFAULT_GRPO_SYSTEM_PROMPT}\n\n"
                    f"User question:\n{question}"
                ),
            }
        ]
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM inference for default GRPO LoRA checkpoints."
    )
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--test-file",
        required=True,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results_vllm_default_grpo")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128, help="vLLM prompt batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Install: pip install vllm transformers peft"
        )

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    adapter_config_path = Path(args.checkpoint_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        lora_rank = int(adapter_config.get("r", 0))
        base_from_adapter = adapter_config.get("base_model_name_or_path", "unknown")
        print(f"Adapter rank (r): {lora_rank}")
        print(f"Adapter base model: {base_from_adapter}")
        if lora_rank > args.max_lora_rank:
            raise ValueError(
                f"LoRA rank {lora_rank} > max_lora_rank {args.max_lora_rank}. "
                "Increase --max-lora-rank."
            )

    tokenizer_path = (
        args.checkpoint_path
        if (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_base_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    llm = LLM(
        model=args.model_base_path,
        tokenizer=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    lora_request = LoRARequest("default_grpo_adapter", 1, args.checkpoint_path)

    test_data = load_split(args.test_file, args.split)

    model_tag = (
        Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    )
    checkpoint_name = Path(args.checkpoint_path).name
    dataset_tag = Path(args.test_file).stem
    output_filename = (
        f"inference_vllm_default_grpo_{model_tag}_{checkpoint_name}_"
        f"{dataset_tag}_seed{args.seed}.json"
    )
    if args.results_suffix:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}{args.results_suffix}{ext}"
    output_path = os.path.join(args.results_dir, output_filename)

    prepared = []
    for idx, item in enumerate(test_data):
        question = item.get("question") or item.get("problem")
        if not question:
            continue
        solution = item.get("solution", "")

        if "answer" in item and item.get("answer") is not None:
            gt_answer_raw = str(item["answer"]).strip()
        else:
            gt_answer_raw = extract_hash_answer(solution)
        if gt_answer_raw is None:
            continue
        try:
            gt_value = float(gt_answer_raw.replace(",", ""))
        except Exception:
            continue

        prepared.append(
            {
                "index": idx,
                "question": question,
                "solution": solution,
                "gt_value": gt_value,
                "prompt": build_prompt(tokenizer, question),
            }
        )

    batch_size = max(1, args.batch_size)
    print(f"Prepared {len(prepared)} examples, batch_size={batch_size}")

    results = []
    correct = 0
    total = 0
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    prompt_tokens_values = []
    completion_tokens_values = []
    total_tokens_values = []
    prompt_tokens_max = 0
    completion_tokens_max = 0
    total_tokens_max = 0

    for start in tqdm(range(0, len(prepared), batch_size), desc="Processing"):
        batch = prepared[start : start + batch_size]
        prompts = [row["prompt"] for row in batch]
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        for row, output in zip(batch, outputs):
            completion = output.outputs[0]
            response = completion.text

            prompt_tokens = len(output.prompt_token_ids or [])
            completion_tokens = len(completion.token_ids or [])
            total_tokens = prompt_tokens + completion_tokens

            prompt_tokens_sum += prompt_tokens
            completion_tokens_sum += completion_tokens
            total_tokens_sum += total_tokens
            prompt_tokens_values.append(prompt_tokens)
            completion_tokens_values.append(completion_tokens)
            total_tokens_values.append(total_tokens)
            prompt_tokens_max = max(prompt_tokens_max, prompt_tokens)
            completion_tokens_max = max(completion_tokens_max, completion_tokens)
            total_tokens_max = max(total_tokens_max, total_tokens)

            gt_value = row["gt_value"]
            model_answer = extract_numeric_from_text(response, force_solution_answer=True)
            model_answer = to_json_safe_real(model_answer)

            is_correct = False
            if model_answer is not None:
                is_correct = abs(model_answer - gt_value) < 1e-6
                if is_correct:
                    correct += 1
            total += 1

            results.append(
                {
                    "index": row["index"],
                    "question": row["question"],
                    "solution": row["solution"],
                    "ground_truth_answer": gt_value,
                    "model_response": response,
                    "model_graph": None,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
            )

    accuracy = correct / total if total > 0 else 0.0
    prompt_mean = prompt_tokens_sum / total if total > 0 else 0.0
    completion_mean = completion_tokens_sum / total if total > 0 else 0.0
    total_mean = total_tokens_sum / total if total > 0 else 0.0
    prompt_tokens_std = (
        float((sum((x - prompt_mean) ** 2 for x in prompt_tokens_values) / total) ** 0.5)
        if total > 0
        else 0.0
    )
    completion_tokens_std = (
        float(
            (sum((x - completion_mean) ** 2 for x in completion_tokens_values) / total)
            ** 0.5
        )
        if total > 0
        else 0.0
    )
    total_tokens_std = (
        float((sum((x - total_mean) ** 2 for x in total_tokens_values) / total) ** 0.5)
        if total > 0
        else 0.0
    )
    prompt_tokens_min = int(min(prompt_tokens_values)) if total > 0 else 0
    completion_tokens_min = int(min(completion_tokens_values)) if total > 0 else 0
    total_tokens_min = int(min(total_tokens_values)) if total > 0 else 0

    payload = {
        "summary": {
            "mode": "default_grpo",
            "model_base_path": args.model_base_path,
            "checkpoint_path": args.checkpoint_path,
            "test_file": args.test_file,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results_folder": output_filename,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "token_usage": {
                "prompt_tokens_total": prompt_tokens_sum,
                "completion_tokens_total": completion_tokens_sum,
                "total_tokens_total": total_tokens_sum,
                "prompt_tokens_avg": prompt_mean,
                "completion_tokens_avg": completion_mean,
                "total_tokens_avg": total_mean,
                "prompt_tokens_std": prompt_tokens_std,
                "completion_tokens_std": completion_tokens_std,
                "total_tokens_std": total_tokens_std,
                "prompt_tokens_min": prompt_tokens_min,
                "completion_tokens_min": completion_tokens_min,
                "total_tokens_min": total_tokens_min,
                "prompt_tokens_max": prompt_tokens_max,
                "completion_tokens_max": completion_tokens_max,
                "total_tokens_max": total_tokens_max,
                "prompt_tokens_range": prompt_tokens_max - prompt_tokens_min,
                "completion_tokens_range": (
                    completion_tokens_max - completion_tokens_min
                ),
                "total_tokens_range": total_tokens_max - total_tokens_min,
            },
        },
        "results": results,
    }

    payload = sanitize_for_json(payload)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)")


if __name__ == "__main__":
    main()
