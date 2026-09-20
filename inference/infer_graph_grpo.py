#!/usr/bin/env python3
"""
Standalone vLLM inference for graph-GRPO LoRA checkpoints.

This script is intended for HF/PEFT LoRA adapters and avoids MLX adapter
compatibility issues.
"""

import argparse
import json
import math
import os
import random
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

GRAPH_SYSTEM_PROMPT = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem, generate a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

Rules:
1. Each constant node has: {"id": <name>, "type": "const", "value": <number>}
2. Each operation node has: {"id": <result_name>, "type": "op", "op": <operation>, "inputs": [<input_ids>]}
3. Available operations: "add", "sub", "mul", "div", "pow", "mod"
4. Operation nodes are named after their result
5. List nodes in computational order (dependencies before usage)
6. Use descriptive names for node IDs
7. Constants should represent actual numbers from the problem
8. Each operation produces a new value
9. Each operation could only use previous const and op nodes as inputs

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]"""


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def parse_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    try:
        return float(value_str)
    except ValueError:
        pass
    if "/" in value_str:
        try:
            return float(Fraction(value_str))
        except Exception:
            pass
    raise ValueError(f"Cannot parse value: {value}")


def extract_graph_from_response(text: str):
    """Return the first balanced JSON array that parses as a list of graph nodes.

    Scans every '[' rather than only the first one, because prose around the
    graph can contain bracket-looking fragments that are not valid JSON.
    """
    for start, ch in enumerate(text):
        if ch != "[":
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except Exception:
                        break
                    if (
                        isinstance(parsed, list)
                        and parsed
                        and all(isinstance(node, dict) for node in parsed)
                    ):
                        return parsed
                    break
    return None


BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "mod": lambda a, b: a % b,
}


UNARY_OPS = {
    "neg": lambda a: -a,
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
    "ln": math.log,
    "log": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


VARIADIC_OPS = {"min": min, "max": max}


def resolve_input(inp, values: dict) -> float:
    """An input slot is either a node id or an inline literal {"value": n}."""
    if isinstance(inp, dict):
        if "value" not in inp:
            raise ValueError(f"Unsupported inline input: {inp}")
        return parse_value(inp["value"])
    key = str(inp).strip()
    if key in values:
        return values[key]
    return parse_value(key)


def execute_graph(nodes) -> float | None:
    """Evaluate the node list in order and return the value of the final node."""
    if not isinstance(nodes, list) or not nodes:
        return None
    try:
        values: dict[str, float] = {}
        last_id = None
        for node in nodes:
            if not isinstance(node, dict):
                return None
            node_id = node.get("id")
            if node_id is None:
                return None
            node_id = str(node_id)
            node_type = node.get("type")

            if node_type == "const":
                values[node_id] = parse_value(node.get("value"))
                last_id = node_id
                continue
            if node_type != "op":
                return None

            op = str(node.get("op", "")).strip().lower()
            args = [resolve_input(x, values) for x in (node.get("inputs") or [])]

            if op in BINARY_OPS and len(args) == 2:
                result = BINARY_OPS[op](*args)
            elif op in BINARY_OPS and len(args) > 2 and op in {"add", "mul"}:
                result = args[0]
                for arg in args[1:]:
                    result = BINARY_OPS[op](result, arg)
            elif op in UNARY_OPS and len(args) == 1:
                result = UNARY_OPS[op](args[0])
            elif op in VARIADIC_OPS and args:
                result = VARIADIC_OPS[op](*args)
            else:
                return None

            values[node_id] = float(result)
            last_id = node_id
        return values.get(last_id) if last_id else None
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


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if has_chat_template:
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
                        f"System instruction:\n{GRAPH_SYSTEM_PROMPT}\n\n"
                        f"User question:\n{question}"
                    ),
                }
            ]
            try:
                return tokenizer.apply_chat_template(
                    fallback_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

    # Final robust fallback for tokenizers without chat_template.
    return (
        f"{GRAPH_SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        "Return only the graph as a JSON list of strings."
    )


def load_split(path, split):
    """Read either a flat list of records or a {"train": [...], "test": [...]} bundle."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if split not in data:
            raise KeyError(f"Split '{split}' not in {path}; available: {sorted(data)}")
        return data[split]
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM inference for graph-GRPO LoRA checkpoints."
    )
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results_vllm")
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
    if not getattr(tokenizer, "chat_template", None) and tokenizer_path != args.model_base_path:
        # LoRA folders may contain tokenizer files without chat_template.
        # Fall back to the base model tokenizer for stable prompting.
        print(
            "WARNING: tokenizer from checkpoint has no chat_template; "
            "falling back to model base tokenizer."
        )
        tokenizer_path = args.model_base_path
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
    lora_request = LoRARequest("sorted_grpo_adapter", 1, args.checkpoint_path)

    test_data = load_split(args.test_file, args.split)

    model_tag = (
        Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    )
    checkpoint_name = Path(args.checkpoint_path).name
    dataset_tag = Path(args.test_file).stem
    output_filename = (
        f"inference_vllm_{model_tag}_{checkpoint_name}_{dataset_tag}_seed{args.seed}.json"
    )
    if args.results_suffix:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}{args.results_suffix}{ext}"
    output_path = os.path.join(args.results_dir, output_filename)

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

    for idx, item in enumerate(tqdm(test_data, desc="Processing")):
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

        prompt = build_prompt(tokenizer, question)
        output = llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )[0]
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

        model_graph = extract_graph_from_response(response)
        model_answer = (
            execute_graph(model_graph) if model_graph else None
        )
        model_answer = to_json_safe_real(model_answer)

        is_correct = False
        if model_answer is not None:
            is_correct = abs(model_answer - gt_value) < 1e-6
            if is_correct:
                correct += 1
        total += 1

        results.append(
            {
                "index": idx,
                "question": question,
                "solution": solution,
                "ground_truth_answer": gt_value,
                "model_response": response,
                "model_graph": model_graph,
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
            "mode": "graph",
            "model_base_path": args.model_base_path,
            "checkpoint_path": args.checkpoint_path,
            "test_file": args.test_file,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results_folder": output_filename,
            "seed": args.seed,
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
