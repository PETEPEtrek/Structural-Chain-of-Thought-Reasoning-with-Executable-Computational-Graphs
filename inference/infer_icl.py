#!/usr/bin/env python3
"""
ICL inference: prompt a base model to emit a computational graph, then execute it.

The graph is a JSON array of nodes, each either a constant
    {"id": "a", "type": "const", "value": 48}
or an operation
    {"id": "b", "type": "op", "op": "mul", "inputs": ["a", "half"]}

The exemplar lives inside the system prompt, so no extra chat turns are needed.
Pass --prompts-json prompts/graph_inference_prompts.json to use the exact
prompt strings from the paper; otherwise identical copies are used from below.

This file has no local imports: it is standalone apart from pip packages.
"""

import argparse
import json
import math
import os
import random
import re
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT_WO_ANSWER = """You are an expert in computational graphs and mathematical reasoning.

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

SYSTEM_PROMPT_W_ANSWER = """You are an expert in computational graphs and mathematical reasoning.

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
10. After the end of the graph, write "ANSWER: <number>"

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]
ANSWER: 72"""

# GSM8k+ redefines +, -, *, / per problem. The prompt therefore asks the model
# to expand every step into primitive nodes and to state the result itself.
SYSTEM_PROMPT_GSM8K_NEW_OPS = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem where the operators +, -, *, / have been REDEFINED, generate a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

The problem text includes the redefined operator definitions. Expand every arithmetic step into primitive nodes (add, sub, mul, div, const) so that standard graph evaluation matches the redefined-operator semantics.

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
10. After the end of the graph, write "ANSWER: <number>"

EXAMPLE 1:
Question: The lunchroom is full of students: 40% are girls and the remainder are boys. There are 2 monitors for every 15 students. There are 8 monitors. Every boy drinks, on average, 1 carton of milk, and every girl drinks, on average, 2 cartons of milk. How many total cartons of milk are consumed by the students in the lunchroom?

---
Note: in this problem the operators +, -, *, / have been REDEFINED. Their standard meanings do NOT apply. On the right-hand side of each definition, standard arithmetic is used.

Definitions:
a + b = 1*(a - b) + 2*(a + b) + -2
a - b = 2*a - 1*b + 6
a * b = 1*a + 4*b + -4
a / b = 4*a + 5*b + 4

Apply these redefined operators in every step of your computation.

[
  {
    "id": "monitors_ratio_denom",
    "type": "const",
    "value": 15
  },
  {
    "id": "monitors_ratio_num",
    "type": "const",
    "value": 2
  },
  {
    "id": "monitors",
    "type": "const",
    "value": 8
  },
  {
    "id": "one",
    "type": "const",
    "value": 1
  },
  {
    "id": "percent_girls",
    "type": "const",
    "value": 0.4
  },
  {
    "id": "milk_per_boy",
    "type": "const",
    "value": 1
  },
  {
    "id": "milk_per_girl",
    "type": "const",
    "value": 2
  },
  {
    "id": "product_ratio",
    "type": "op",
    "op": "mul",
    "inputs": [
      "monitors_ratio_denom",
      "monitors_ratio_num"
    ]
  },
  {
    "id": "percent_boys",
    "type": "op",
    "op": "sub",
    "inputs": [
      "one",
      "percent_girls"
    ]
  },
  {
    "id": "students_per_monitor",
    "type": "op",
    "op": "div",
    "inputs": [
      "product_ratio",
      "monitors_ratio_num"
    ]
  },
  {
    "id": "total_students",
    "type": "op",
    "op": "mul",
    "inputs": [
      "students_per_monitor",
      "monitors"
    ]
  },
  {
    "id": "boys",
    "type": "op",
    "op": "mul",
    "inputs": [
      "total_students",
      "percent_boys"
    ]
  },
  {
    "id": "girls",
    "type": "op",
    "op": "mul",
    "inputs": [
      "total_students",
      "percent_girls"
    ]
  },
  {
    "id": "boys_milk",
    "type": "op",
    "op": "mul",
    "inputs": [
      "boys",
      "milk_per_boy"
    ]
  },
  {
    "id": "girls_milk",
    "type": "op",
    "op": "mul",
    "inputs": [
      "girls",
      "milk_per_girl"
    ]
  },
  {
    "id": "total_milk",
    "type": "op",
    "op": "add",
    "inputs": [
      "boys_milk",
      "girls_milk"
    ]
  }
]
ANSWER: 550.8

EXAMPLE 2:
Question: Maia is a freelance blogger working for different clients for whom she writes different articles every day. She gets 6 client requests every day and works on four of them each day. How many client requests will she have remaining to work on after 5 days.

---
Note: in this problem the operators +, -, *, / have been REDEFINED. Their standard meanings do NOT apply. On the right-hand side of each definition, standard arithmetic is used.

Definitions:
a + b = 2*(a - b) + 1*(a + b) + 2
a - b = 3*a + 1*b + -3
a * b = 3*a - 3*b + 0
a / b = 4*a - 2*b + 3

Apply these redefined operators in every step of your computation.

[
  {
    "id": "requests_per_day",
    "type": "const",
    "value": 6
  },
  {
    "id": "worked_requests_per_day",
    "type": "const",
    "value": 4
  },
  {
    "id": "days",
    "type": "const",
    "value": 5
  },
  {
    "id": "remaining_per_day",
    "type": "op",
    "op": "sub",
    "inputs": [
      "requests_per_day",
      "worked_requests_per_day"
    ]
  },
  {
    "id": "remaining_total",
    "type": "op",
    "op": "mul",
    "inputs": [
      "days",
      "remaining_per_day"
    ]
  }
]
ANSWER: -42"""

# Datasets whose answer is read from the ANSWER tag instead of graph execution,
# because their operators are redefined and standard execution is meaningless.
ANSWER_TAG_DATASETS = {"gsm8k_new_ops"}

DATASET_CHOICES = ["gsm8k", "gsm8k_new_ops", "igsm", "math"]


def select_system_prompt(dataset_name: str, prompts_json: str | None) -> str:
    if prompts_json:
        data = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        examples = data.get("examples", {})
        if dataset_name in examples:
            return examples[dataset_name]
        return data["prompts"]["SYSTEM_PROMPT_W_ANSWER"]
    if dataset_name == "gsm8k_new_ops":
        return SYSTEM_PROMPT_GSM8K_NEW_OPS
    return SYSTEM_PROMPT_W_ANSWER


# --------------------------------------------------------------------------- #
# Graph parsing and execution
# --------------------------------------------------------------------------- #

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


def parse_value(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


def load_split(path, split):
    """Read either a flat list of records or a {"train": [...], "test": [...]} bundle."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if split not in data:
            raise KeyError(f"Split '{split}' not in {path}; available: {sorted(data)}")
        return data[split]
    return data


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def extract_answer_tag_number(text: str):
    match = re.search(r"ANSWER:\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except Exception:
        return None


def get_question(example: dict):
    return example.get("question") or example.get("problem")


def get_gt_answer(example: dict):
    if example.get("answer") is not None:
        return str(example["answer"]).strip()
    return extract_hash_answer(example.get("solution", ""))


def to_json_safe_real(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, complex):
        if value.imag != 0.0:
            return None
        value = value.real
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, complex):
        return None
    return to_json_safe_real(value)


def is_gemma_2_2b(path_like: str) -> bool:
    value = path_like.lower()
    return "gemma-2-2b" in value or "gemma_2_2b" in value


def build_prompt(tokenizer, system_prompt: str, question: str, allow_system_role: bool) -> str:
    if allow_system_role:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        # Gemma-2 templates reject a system role; fold it into the user turn.
        messages = [
            {
                "role": "user",
                "content": f"System instruction:\n{system_prompt}\n\n{question}",
            }
        ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="ICL inference for computational graph generation."
    )
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--dataset-name", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="prompts/graph_inference_prompts.json, to reproduce the paper prompts exactly",
    )
    parser.add_argument("--results-dir", default="results_icl")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("Missing dependencies. Install: pip install vllm transformers")

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_data = load_split(args.test_file, args.split)
    use_answer_tag = args.dataset_name in ANSWER_TAG_DATASETS
    system_prompt = select_system_prompt(args.dataset_name, args.prompts_json)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base_path, trust_remote_code=args.trust_remote_code
    )
    allow_system_role = not is_gemma_2_2b(args.model_base_path)

    llm = LLM(
        model=args.model_base_path,
        tokenizer=args.model_base_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens
    )

    os.makedirs(args.results_dir, exist_ok=True)
    model_tag = Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    output_filename = f"inference_icl_{model_tag}_{args.dataset_name}_seed{args.seed}.json"
    if args.results_suffix:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}{args.results_suffix}{ext}"
    output_path = os.path.join(args.results_dir, output_filename)

    results = []
    total = 0
    correct = 0

    for idx, item in enumerate(tqdm(test_data, desc=f"Processing {args.dataset_name}")):
        question = get_question(item)
        if not question:
            continue
        gt_raw = get_gt_answer(item)
        if gt_raw is None:
            continue
        try:
            gt_value = float(str(gt_raw).replace(",", ""))
        except Exception:
            continue

        prompt = build_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            question=question,
            allow_system_role=allow_system_role,
        )
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        response = output.outputs[0].text

        model_graph = extract_graph_from_response(response)
        graph_value = to_json_safe_real(execute_graph(model_graph) if model_graph else None)
        if use_answer_tag:
            model_answer = extract_answer_tag_number(response)
            tolerance = 1e-4
        else:
            model_answer = graph_value
            tolerance = 1e-6
        model_answer = to_json_safe_real(model_answer)

        is_correct = model_answer is not None and abs(model_answer - gt_value) < tolerance
        correct += int(is_correct)
        total += 1

        results.append(
            {
                "index": idx,
                "question": question,
                "redefined_ops": item.get("redefined_ops"),
                "ground_truth_answer": gt_value,
                "model_response": response,
                "model_graph": model_graph,
                "graph_value": graph_value,
                "model_answer": model_answer,
                "is_correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    payload = {
        "summary": {
            "mode": "icl_graph",
            "model_base_path": args.model_base_path,
            "dataset_name": args.dataset_name,
            "test_file": args.test_file,
            "split": args.split,
            "prompts_json": args.prompts_json,
            "answer_source": "answer_tag" if use_answer_tag else "graph_execution",
            "seed": args.seed,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results_file": output_filename,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(payload), f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)")


if __name__ == "__main__":
    main()
