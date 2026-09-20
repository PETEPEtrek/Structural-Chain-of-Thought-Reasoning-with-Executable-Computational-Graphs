#!/usr/bin/env python3
"""
Configurable GRPO training for computational graph generation.

Behavior inferred from checkpoint name:
- full / not_full
- dataset family: gsm8k / igsm / math / gsm8k_new_ops
- has_answer / has_no_answer
"""

import argparse
import ast
import json
import math
import os
import random
import re
from collections import Counter
from fractions import Fraction
from typing import List

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, activations, set_seed
from trl import GRPOConfig, GRPOTrainer

import graph_prompts
from graph_exec import computational_graph_to_lines, safe_eval_math_expr


activations.PytorchGELUTanh = activations.GELUTanh

DEFAULT_MODEL_NAME = "models/petepetrek/Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SFT_CHECKPOINT = (
    "outputs_sft_graphs_has_no_answer/not_full_lora/pre_grpo"
)

DATASET_FILE_MAP = {
    "gsm8k": "gsm8k_comp_graphs_sft_grpo.json",
    "igsm": "igsm_openrouter_valid_train_grpo.json",
    "math": "math_openrouter_valid_train_grpo.json",
    "gsm8k_new_ops": "gsm8k_new_ops_comp_graphs_sft_grpo.json",
}

OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
    "mod": "%",
}


def sanitize_name(value: str) -> str:
    value = value.lower().replace("-", "_").replace("/", "_").replace(".", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


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


def normalize_expr(expr: str) -> str:
    expr = expr.strip()
    if "@" in expr:
        expr = expr.split("@", 1)[0].strip()
    return expr


def eval_rhs(expr: str, values: dict) -> float:
    """Evaluate one assignment RHS: plain binops first, then the wider math grammar."""
    expr_n = normalize_expr(expr)
    last_err = None
    try:
        return float(parse_value(expr_n))
    except Exception as e:
        last_err = e
    try:
        return float(safe_eval_expr(expr_n, values))
    except Exception as e:
        last_err = e
    try:
        return float(safe_eval_math_expr(expr_n, values))
    except Exception as e:
        last_err = e
    raise ValueError(f"cannot eval {expr_n!r}: {last_err}")


def safe_eval_expr(expr: str, values: dict):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        raise ValueError("Unsupported expression node")

    return _eval(ast.parse(expr, mode="eval"))


def convert_dict_graph_to_lines(comp_graph):
    transformed = []
    for node in comp_graph:
        if not isinstance(node, dict):
            return None
        if node.get("type") == "const":
            transformed.append(f"{node['id']}={node['value']}")
        elif node.get("type") == "op":
            op = OP_SYMBOLS.get(node.get("op"), node.get("op", ""))
            inputs = node.get("inputs", [])
            expr = op.join(inputs) if inputs else "0"
            transformed.append(f"{node['id']}={expr}")
    return transformed


def normalize_to_lines(graph_obj):
    if not isinstance(graph_obj, list):
        return None
    if not graph_obj:
        return []
    first = graph_obj[0]
    if isinstance(first, str):
        if all(isinstance(node, str) and "=" in node for node in graph_obj):
            return graph_obj
        return None
    if isinstance(first, dict):
        lines = computational_graph_to_lines(graph_obj)
        if lines is not None:
            return lines
        return convert_dict_graph_to_lines(graph_obj)
    return None


def parse_graph_from_response(response: str):
    start = response.find("[")
    if start == -1:
        # Fallback: some models output assignment lines directly
        # without wrapping them into a JSON array.
        candidate_lines = []
        for raw_line in response.splitlines():
            line = raw_line.strip().strip("`").strip().rstrip(",")
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()
            if not left or not right:
                continue
            candidate_lines.append(f"{left}={right}")
        return candidate_lines if candidate_lines else None
    bracket_count = 0
    for idx in range(start, len(response)):
        if response[idx] == "[":
            bracket_count += 1
        elif response[idx] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                try:
                    return json.loads(response[start : idx + 1])
                except Exception:
                    break
    # Fallback if bracketed parse fails.
    candidate_lines = []
    for raw_line in response.splitlines():
        line = raw_line.strip().strip("`").strip().rstrip(",")
        if not line or "=" not in line:
            continue
        left, right = line.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        candidate_lines.append(f"{left}={right}")
    if candidate_lines:
        return candidate_lines
    return None


def compute_from_lines(graph_lines):
    try:
        values = {}
        last_node = None
        for line in graph_lines:
            if not isinstance(line, str) or "=" not in line:
                return None
            node_id, expr = line.split("=", 1)
            node_id = node_id.strip()
            expr = normalize_expr(expr)
            if not node_id:
                return None
            try:
                value = eval_rhs(expr, values)
            except Exception:
                return None
            values[node_id] = value
            last_node = node_id
        if last_node is None:
            return None
        return values.get(last_node)
    except Exception:
        return None


def compute_node_values(graph_lines):
    try:
        values = {}
        const_values = []
        op_values = []
        for line in graph_lines:
            if not isinstance(line, str) or "=" not in line:
                return None, None
            node_id, expr = line.split("=", 1)
            node_id = node_id.strip()
            expr = normalize_expr(expr)
            try:
                value = parse_value(expr)
                const_values.append(value)
            except Exception:
                try:
                    value = eval_rhs(expr, values)
                except Exception:
                    return None, None
                op_values.append(value)
            values[node_id] = value
        return const_values, op_values
    except Exception:
        return None, None


def count_value_intersection(values_a, values_b, precision=6):
    def to_key(value):
        return f"{round(value, precision):.{precision}f}"

    counter_a = Counter(to_key(v) for v in values_a)
    counter_b = Counter(to_key(v) for v in values_b)
    return sum((counter_a & counter_b).values())


def sample_const_nodes(graph_lines, sampler_func):
    sampled = []
    for line in graph_lines:
        if "=" not in line:
            continue
        node_id, expr = line.split("=", 1)
        expr = normalize_expr(expr)
        try:
            parse_value(expr)
            sampled.append(f"{node_id.strip()}={sampler_func()}")
        except Exception:
            sampled.append(line)
    return sampled


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_answer_block_value(response: str):
    match = re.search(
        r"ANSWER:\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value_str = match.group(1).replace(",", "")
    try:
        return parse_value(value_str)
    except Exception:
        return None


def get_question_text(item: dict, dataset_name: str) -> str:
    if dataset_name == "igsm":
        # OpenRouter IGSM files use "question"; keep backward compatibility.
        return item.get("question") or item.get("problem", "")
    return item.get("question", "")


def get_target_answer(item: dict, dataset_name: str):
    if dataset_name in {"igsm", "math", "gsm8k_new_ops"}:
        return item.get("answer")
    return extract_hash_answer(item.get("solution", ""))


def infer_checkpoint_flags(checkpoint_path: str):
    name = checkpoint_path.lower()
    use_full = "full" in name and "not_full" not in name

    if "has_no_answer" in name:
        has_answer = False
    elif "has_answer" in name:
        has_answer = True
    else:
        has_answer = False

    if "gsm8k_new_ops" in name:
        dataset_name = "gsm8k_new_ops"
    elif "igsm" in name:
        dataset_name = "igsm"
    elif "math" in name:
        dataset_name = "math"
    else:
        dataset_name = "gsm8k"
    return use_full, has_answer, dataset_name


def select_graph_system_prompt(use_full: bool, has_answer: bool) -> str:
    if use_full and has_answer:
        return graph_prompts.SYSTEM_PROMPT_FULL_W_ANSWER
    if use_full and not has_answer:
        return graph_prompts.SYSTEM_PROMPT_FULL_WO_ANSWER
    if has_answer:
        return graph_prompts.SYSTEM_PROMPT_W_ANSWER
    return graph_prompts.SYSTEM_PROMPT


def get_graph_payload(item: dict):
    return item.get("computational_graph", [])


def build_training_dataset(
    json_file: str,
    system_prompt: str,
    dataset_scheduler: str,
    dataset_name: str,
    seed: int,
    allow_system_role: bool = True,
) -> Dataset:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    def sort_key(item):
        graph_len = len(get_graph_payload(item))
        solution_len = len(item.get("solution", "") or "")
        return (graph_len, solution_len)

    if dataset_scheduler == "sorted":
        data.sort(key=sort_key)

    dataset_data = []
    for item in data:
        graph_payload = get_graph_payload(item)
        graph_json = json.dumps(graph_payload)
        question_text = get_question_text(item, dataset_name)
        target_answer = get_target_answer(item, dataset_name)
        if not question_text or target_answer is None:
            continue
        if allow_system_role:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]
        else:
            prompt = [
                {
                    "role": "user",
                    "content": (
                        f"System instruction:\n{system_prompt}\n\n"
                        f"User question:\n{question_text}"
                    ),
                }
            ]
        dataset_data.append(
            {
                "prompt": prompt,
                "graph": graph_json,
                "answer": target_answer,
            }
        )
    if not dataset_data:
        raise ValueError(
            "GRPO dataset is empty after preprocessing. "
            f"dataset_name={dataset_name}, json_file={json_file}"
        )
    dataset = Dataset.from_list(dataset_data)
    if dataset_scheduler == "default":
        dataset = dataset.shuffle(seed=seed)
    return dataset


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    return os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json"))


def build_reward_funcs(reward_names: List[str]):
    def style_reward_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            response = completion[0].get("content", "")
            try:
                graph_obj = parse_graph_from_response(response)
                if graph_obj is None:
                    rewards.append(-2.0)
                    continue
                ok = (
                    isinstance(graph_obj, list)
                    and len(graph_obj) > 0
                    and all(isinstance(node, dict) for node in graph_obj)
                )
                rewards.append(2.0 if ok else -2.0)
            except Exception:
                rewards.append(-2.0)
        return rewards

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        rewards = []
        for completion, true_answer in zip(completions, answer):
            response = completion[0].get("content", "")
            try:
                graph_obj = parse_graph_from_response(response)
                graph_lines = normalize_to_lines(graph_obj)
                if graph_lines is None:
                    rewards.append(-2.0)
                    continue
                computed_value = compute_from_lines(graph_lines)
                if computed_value is None:
                    rewards.append(-1.5)
                    continue
                try:
                    true_value = float(str(true_answer).replace(",", ""))
                except Exception:
                    rewards.append(0.0)
                    continue
                rewards.append(3.0 if abs(computed_value - true_value) < 1e-6 else -0.5)
            except Exception:
                rewards.append(-1.5)
        return rewards

    def answer_reward_func(completions, answer, **kwargs) -> List[float]:
        rewards = []
        for completion, true_answer in zip(completions, answer):
            response = completion[0].get("content", "")
            pred_value = extract_answer_block_value(response)
            if pred_value is None:
                rewards.append(-2.0)
                continue
            try:
                true_value = float(str(true_answer).replace(",", ""))
            except Exception:
                rewards.append(0.0)
                continue
            rewards.append(3.0 if abs(pred_value - true_value) < 1e-6 else -0.5)
        return rewards

    def overlap_reward_func(completions, graph, **kwargs) -> List[float]:
        rewards = []
        for completion, gold_graph_data in zip(completions, graph):
            response = completion[0].get("content", "")
            try:
                gen_graph_obj = parse_graph_from_response(response)
                gen_graph = normalize_to_lines(gen_graph_obj)
                gold_graph_obj = (
                    json.loads(gold_graph_data)
                    if isinstance(gold_graph_data, str)
                    else gold_graph_data
                )
                gold_graph = normalize_to_lines(gold_graph_obj)
                if gold_graph is None or gen_graph is None:
                    rewards.append(-2.0)
                    continue
                gold_const, gold_op = compute_node_values(gold_graph)
                gen_const, gen_op = compute_node_values(gen_graph)
                if gold_const is None or gen_const is None:
                    rewards.append(-1.5)
                    continue
                const_ratio = count_value_intersection(gold_const, gen_const) / max(
                    len(gold_const), 1
                )
                op_ratio = count_value_intersection(gold_op, gen_op) / max(len(gold_op), 1)
                rewards.append(const_ratio + op_ratio)
            except Exception:
                rewards.append(-1.5)
        return rewards

    def minimality_reward_func(completions, lambda_coef=0.05, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            response = completion[0].get("content", "")
            try:
                graph_obj = parse_graph_from_response(response)
                graph_lines = normalize_to_lines(graph_obj)
                if graph_lines is None:
                    rewards.append(0.0)
                    continue
                rewards.append(math.exp(-lambda_coef * len(graph_lines)))
            except Exception:
                rewards.append(0.0)
        return rewards

    def no_redundancy_reward_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            response = completion[0].get("content", "")
            try:
                graph_obj = parse_graph_from_response(response)
                graph_lines = normalize_to_lines(graph_obj)
                if not graph_lines:
                    rewards.append(0.0)
                    continue
                node_ids = [line.split("=", 1)[0].strip() for line in graph_lines]
                total = len(node_ids)
                if total == 0:
                    rewards.append(0.0)
                    continue
                duplicates = total - len(set(node_ids))
                rewards.append(-(duplicates / total))
            except Exception:
                rewards.append(0.0)
        return rewards

    def structural_completeness_reward_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            response = completion[0].get("content", "")
            try:
                graph_obj = parse_graph_from_response(response)
                graph_lines = normalize_to_lines(graph_obj)
                if not graph_lines:
                    rewards.append(0.0)
                    continue
                node_ids = []
                used_vars = set()
                for line in graph_lines:
                    if "=" not in line:
                        continue
                    node_id, expr = line.split("=", 1)
                    node_id = node_id.strip()
                    expr = normalize_expr(expr)
                    if not node_id:
                        continue
                    node_ids.append(node_id)
                    try:
                        parsed = ast.parse(expr, mode="eval")
                        for node in ast.walk(parsed):
                            if isinstance(node, ast.Name):
                                used_vars.add(node.id)
                    except Exception:
                        continue
                if not node_ids:
                    rewards.append(0.0)
                    continue
                intermediate = node_ids[:-1]
                if not intermediate:
                    rewards.append(1.0)
                    continue
                used_count = sum(1 for node_id in intermediate if node_id in used_vars)
                rewards.append(used_count / len(intermediate))
            except Exception:
                rewards.append(0.0)
        return rewards

    def monte_carlo_reward_func(completions, graph, trials=100, **kwargs) -> List[float]:
        rewards = []
        for completion, gold_graph_data in zip(completions, graph):
            response = completion[0].get("content", "")
            try:
                gen_graph_obj = parse_graph_from_response(response)
                gen_graph = normalize_to_lines(gen_graph_obj)
                gold_graph_obj = (
                    json.loads(gold_graph_data)
                    if isinstance(gold_graph_data, str)
                    else gold_graph_data
                )
                gold_graph = normalize_to_lines(gold_graph_obj)
                if gold_graph is None or gen_graph is None:
                    rewards.append(-2.0)
                    continue

                matches = 0
                for _ in range(trials):
                    sampled_value = random.uniform(-100, 100)

                    def sampler():
                        return sampled_value

                    sampled_gold = sample_const_nodes(gold_graph, sampler)
                    sampled_gen = sample_const_nodes(gen_graph, sampler)
                    gold_result = compute_from_lines(sampled_gold)
                    gen_result = compute_from_lines(sampled_gen)
                    if gold_result is None or gen_result is None:
                        continue
                    if not (math.isfinite(gold_result) and math.isfinite(gen_result)):
                        continue
                    if abs(gold_result - gen_result) < 1e-6:
                        matches += 1

                ratio = matches / max(trials, 1)
                if ratio >= 0.95:
                    rewards.append(2.0)
                elif ratio >= 0.8:
                    rewards.append(1.0)
                elif ratio >= 0.5:
                    rewards.append(0.5)
                else:
                    rewards.append(-0.5)
            except Exception:
                rewards.append(-2.0)
        return rewards

    reward_map = {
        "style": style_reward_func,
        "correctness": correctness_reward_func,
        "answer_reward": answer_reward_func,
        "overlap": overlap_reward_func,
        "minimality": minimality_reward_func,
        "no_redundancy": no_redundancy_reward_func,
        "structural_completeness": structural_completeness_reward_func,
        "monte_carlo": monte_carlo_reward_func,
    }
    funcs = []
    for name in reward_names:
        key = name.strip()
        if key not in reward_map:
            raise ValueError(f"Unknown reward: {key}")
        funcs.append(reward_map[key])
    return funcs


def resolve_paths(
    model_name: str,
    dataset_scheduler: str,
    max_steps: int,
    reward_names: List[str],
    num_generations: int,
    dataset_name: str,
    use_full: bool,
    seed: int,
    save_path: str | None,
    output_dir: str | None,
    logging_dir: str | None,
):
    model_tag = sanitize_name(model_name.split("/")[-1] or model_name)
    rewards_tag = "_".join(reward_names)
    full_tag = "full" if use_full else "not_full"
    default_save_path = (
        f"{model_tag}_{dataset_name}_{dataset_scheduler}_scheduler_{max_steps}_"
        f"{rewards_tag}_{num_generations}_{full_tag}_seed{seed}"
    )
    final_save_path = save_path or default_save_path
    final_output_dir = output_dir or final_save_path
    final_logging_dir = logging_dir or f"runs_{final_save_path}"
    return final_save_path, final_output_dir, final_logging_dir


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training script with checkpoint-based auto config.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sft-checkpoint", default=DEFAULT_SFT_CHECKPOINT)
    parser.add_argument("--dataset-file")
    parser.add_argument(
        "--dataset-scheduler",
        choices=["default", "sorted"],
        default="default",
    )
    parser.add_argument(
        "--rewards",
        default="style,correctness,overlap",
        help=(
            "Comma-separated reward names from: "
            "style,correctness,answer_reward,overlap,minimality,no_redundancy,"
            "structural_completeness,monte_carlo"
        ),
    )
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=600)
    parser.add_argument("--save-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--logging-dir")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    reward_names = [x.strip() for x in args.rewards.split(",") if x.strip()]

    use_full, has_answer, dataset_name = infer_checkpoint_flags(args.sft_checkpoint)
    dataset_file = args.dataset_file or DATASET_FILE_MAP[dataset_name]
    system_prompt = select_graph_system_prompt(use_full=use_full, has_answer=has_answer)

    reward_funcs = build_reward_funcs(reward_names=reward_names)
    save_path, output_dir, logging_dir = resolve_paths(
        model_name=args.model_name,
        dataset_scheduler=args.dataset_scheduler,
        max_steps=args.max_steps,
        reward_names=reward_names,
        num_generations=args.num_generations,
        dataset_name=dataset_name,
        use_full=use_full,
        seed=args.seed,
        save_path=args.save_path,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    denom = 4 * args.per_device_train_batch_size
    if denom <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    gradient_accumulation_steps = max(1, math.ceil(args.num_generations / denom))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_lora_checkpoint(args.sft_checkpoint):
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.sft_checkpoint,
            is_trainable=True,
            adapter_name="default",
        )
        model.set_adapter("default")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    dataset = build_training_dataset(
        json_file=dataset_file,
        system_prompt=system_prompt,
        dataset_scheduler=args.dataset_scheduler,
        dataset_name=dataset_name,
        seed=args.seed,
        allow_system_role=not is_gemma_2_2b(args.model_name),
    )
    print(f"  train_examples: {len(dataset)}")

    print("Resolved config:")
    print(f"  checkpoint: {args.sft_checkpoint}")
    print(f"  dataset_name: {dataset_name}")
    print(f"  dataset_file: {dataset_file}")
    print(f"  full: {use_full}")
    print(f"  has_answer: {has_answer}")
    print(f"  output_dir: {output_dir}")
    print(f"  logging_dir: {logging_dir}")
    print(f"  save_path: {save_path}")

    training_args = GRPOConfig(
        learning_rate=5e-6,
        use_liger_kernel=True,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="tensorboard",
        logging_dir=logging_dir,
        output_dir=output_dir,
        fp16=True,
        bf16=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
