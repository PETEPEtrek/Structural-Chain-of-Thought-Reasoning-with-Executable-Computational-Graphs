#!/usr/bin/env python3
"""
Intervention-based faithfulness for executable computational graphs.

Metrics (per sample, then averaged over the samples where they are defined):

  F_ID      the emitted answer equals exec(generated_graph)
  F_OOD     after replacing the graph with an intervention graph, the newly
            emitted answer equals exec(intervention_graph)
  F_strong  F_ID and F_OOD both hold

Intervention protocol (COR_if_incorrect_else_CNF):

  CNF  the generated graph already reproduces the gold answer, so it is mutated
       (one constant or one operator) until it no longer does
  COR  the generated graph is wrong, so it is replaced by the gold graph

Because F_strong is only defined where both F_ID and F_OOD are defined, the
plain F_OOD average uses a larger denominator and can end up below F_strong.
F_OOD_aligned re-averages F_OOD over exactly the samples where F_strong is
defined, which is the comparable number to report next to F_strong.

Modes:
  offline  recompute the metrics from an inference JSON that already contains
           generated graphs, answers and interventions
  online   run vLLM: generate graph+ANSWER, build the intervention, then ask
           the model to continue an ANSWER for the intervention graph

This file has no local imports: it is standalone apart from pip packages.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

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

SYSTEM_PROMPT_WO_ANSWER = SYSTEM_PROMPT_W_ANSWER.replace(
    '9. Each operation could only use previous const and op nodes as inputs\n'
    '10. After the end of the graph, write "ANSWER: <number>"',
    "9. Each operation could only use previous const and op nodes as inputs",
).replace("\nANSWER: 72", "")

GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] | trim + '\n\n' %}"
    "{% set messages = messages[1:] %}"
    "{% else %}{% set system_message = '' %}{% endif %}"
    "{% for message in messages %}"
    "{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + content | trim + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

# Datasets whose gold answer lives in item["answer"] instead of the #### tail.
ANSWER_FIELD_DATASETS = {"igsm", "math", "gsm8k_new_ops", "math_new_ops"}
REDEFINED_OPS_DATASETS = {"gsm8k_new_ops", "math_new_ops"}

INFIX_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**", "mod": "%"}
OP_PATTERN = re.compile(r"\*\*|[+\-*/%]")
OP_SYMBOLS = ["+", "-", "*", "/", "**", "%"]
CONST_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")

TOLERANCE = 1e-4


# --------------------------------------------------------------------------- #
# Graph parsing and execution
# --------------------------------------------------------------------------- #


def parse_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    if "/" in text:
        try:
            return float(Fraction(text))
        except Exception:
            pass
    raise ValueError(f"Cannot parse value: {value!r}")


_EVAL_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "min": min,
    "max": max,
}

_AST_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
}


def safe_eval_expr(expr: str, values: Dict[str, float]) -> float:
    """Evaluate an assignment right-hand side without using eval()."""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return float(node.value)
            raise ValueError("unsupported constant")
        if isinstance(node, ast.BinOp):
            handler = _AST_BINOPS.get(type(node.op))
            if handler is None:
                raise ValueError("unsupported operator")
            return handler(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("unsupported call target")
            func = _EVAL_FUNCS.get(node.func.id.lower())
            if func is None:
                raise ValueError(f"unsupported function {node.func.id}")
            return func(*[_eval(a) for a in node.args])
        raise ValueError("unsupported expression node")

    return float(_eval(ast.parse(expr, mode="eval")))


def normalize_expr(expr: str) -> str:
    expr = expr.strip()
    if "@" in expr:
        expr = expr.split("@", 1)[0].strip()
    return expr


def compute_from_lines(graph_lines: List[str]) -> Optional[float]:
    """Execute assignment lines like ['a=3', 'b=a*2'] and return the last value."""
    try:
        values: Dict[str, float] = {}
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
                value = parse_value(expr)
            except Exception:
                value = safe_eval_expr(expr, values)
            values[node_id] = value
            last_node = node_id
        return values.get(last_node) if last_node else None
    except Exception:
        return None


def apply_redefined(op: str, a: float, b: float, redefined_ops: Dict[str, str]) -> float:
    """Evaluate one redefined binary operator; its RHS uses standard arithmetic."""
    return safe_eval_expr(redefined_ops[op], {"a": float(a), "b": float(b)})


def eval_rhs_redefined(
    expr: str, values: Dict[str, float], redefined_ops: Dict[str, str]
) -> float:
    expr = normalize_expr(expr).replace(" ", "")
    parts = [p for p in re.split(r"(\*\*|[+\-*/])", expr) if p]

    def resolve(token: str) -> float:
        try:
            return parse_value(token)
        except ValueError:
            if token not in values:
                raise KeyError(token)
            return float(values[token])

    if len(parts) == 1:
        return resolve(parts[0])
    result = resolve(parts[0])
    index = 1
    while index < len(parts):
        result = apply_redefined(parts[index], result, resolve(parts[index + 1]), redefined_ops)
        index += 2
    return result


def compute_from_lines_redefined(
    graph_lines: List[str], redefined_ops: Dict[str, str]
) -> Optional[float]:
    try:
        values: Dict[str, float] = {}
        last_node = None
        for line in graph_lines:
            if not isinstance(line, str) or "=" not in line:
                return None
            node_id, expr = line.split("=", 1)
            node_id = node_id.strip()
            expr = expr.strip()
            if not node_id:
                return None
            if re.fullmatch(r"-?\d+(?:\.\d+)?", expr):
                values[node_id] = float(expr)
            else:
                values[node_id] = eval_rhs_redefined(expr, values, redefined_ops)
            last_node = node_id
        return values.get(last_node) if last_node else None
    except Exception:
        return None


def _input_atom(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "1.0" if value else "0.0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 10**15:
            return str(int(round(value)))
        return repr(value)
    if isinstance(value, dict) and "value" in value:
        return _input_atom(value["value"])
    raise ValueError(f"unsupported graph input: {value!r}")


def graph_to_lines(graph: List[dict]) -> Optional[List[str]]:
    """Convert [{'id':..,'type':'op','op':'mul','inputs':[..]}] to line format."""
    if not isinstance(graph, list) or not graph:
        return None
    aliases = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}
    lines: List[str] = []
    for node in graph:
        if not isinstance(node, dict) or node.get("id") is None:
            return None
        node_id = str(node["id"])
        node_type = node.get("type")

        if node_type == "const":
            if "value" not in node:
                return None
            lines.append(f"{node_id}={_input_atom(node['value'])}")
            continue
        if node_type != "op":
            return None

        op = str(node.get("op", "")).strip().lower()
        try:
            atoms = [_input_atom(x) for x in (node.get("inputs") or [])]
        except ValueError:
            return None

        if op == "neg" and len(atoms) == 1:
            rhs = f"-{atoms[0]}"
        elif len(atoms) == 1:
            rhs = f"{aliases.get(op, op)}({atoms[0]})"
        elif len(atoms) == 2 and op in INFIX_OPS:
            rhs = f"{atoms[0]}{INFIX_OPS[op]}{atoms[1]}"
        elif op in {"mul", "add"} and len(atoms) >= 2:
            rhs = INFIX_OPS[op].join(atoms)
        elif op in {"min", "max"} and len(atoms) >= 2:
            rhs = f"{op}(" + ",".join(atoms) + ")"
        else:
            rhs = f"{aliases.get(op, op)}(" + ",".join(atoms) + ")"
        lines.append(f"{node_id}={rhs}")
    return lines


def normalize_to_lines(graph_obj: Any) -> Optional[List[str]]:
    if not isinstance(graph_obj, list):
        return None
    if not graph_obj:
        return []
    if isinstance(graph_obj[0], str):
        if all(isinstance(node, str) and "=" in node for node in graph_obj):
            return list(graph_obj)
        return None
    if isinstance(graph_obj[0], dict):
        try:
            return graph_to_lines(graph_obj)
        except (ValueError, TypeError):
            return None
    return None


def parse_graph_from_response(response: str) -> Optional[List[Any]]:
    """Prefer a JSON array; otherwise collect bare `name=expr` lines."""

    def bare_lines() -> Optional[List[str]]:
        collected = []
        for raw in response.splitlines():
            line = raw.strip().strip("`").strip().rstrip(",")
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            if left.strip() and right.strip():
                collected.append(f"{left.strip()}={right.strip()}")
        return collected or None

    start = response.find("[")
    if start == -1:
        return bare_lines()
    depth = 0
    for index in range(start, len(response)):
        if response[index] == "[":
            depth += 1
        elif response[index] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response[start : index + 1])
                except Exception:
                    break
    return bare_lines()


def extract_answer_tag_number(text: str) -> Optional[float]:
    match = re.search(r"ANSWER:\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", text, flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except Exception:
        return None


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def to_json_safe_real(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return None


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    return to_json_safe_real(value)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #


def infer_dataset_name(test_file: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    name = Path(test_file).name.lower()
    if "math_new_ops" in name:
        return "math_new_ops"
    if "new_ops" in name:
        return "gsm8k_new_ops"
    if "igsm" in name:
        return "igsm"
    if "math" in name:
        return "math"
    return "gsm8k"


def load_split(path: str, split: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in (split, "test", "valid", "validation"):
            if key in data:
                return data[key]
        raise ValueError(f"Could not find split '{split}' in {path}")
    return data


def get_question_text(item: dict) -> str:
    return item.get("question") or item.get("problem") or ""


def gold_answer_value(item: dict, dataset_name: str) -> Optional[float]:
    if dataset_name in ANSWER_FIELD_DATASETS:
        raw = item.get("answer")
    else:
        raw = extract_hash_answer(item.get("solution", "") or "")
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", ""))
    except Exception:
        return None


def gold_graph_lines(item: dict) -> Optional[List[str]]:
    transformed = item.get("computational_graph_transformed")
    if isinstance(transformed, list) and transformed:
        if all(isinstance(x, str) for x in transformed):
            return list(transformed)
    graph = item.get("computational_graph")
    return normalize_to_lines(graph) if graph is not None else None


def exec_lines(
    lines: Optional[List[str]], item: dict, dataset_name: str
) -> Optional[float]:
    if not lines:
        return None
    redefined = item.get("redefined_ops")
    if dataset_name in REDEFINED_OPS_DATASETS and redefined:
        return compute_from_lines_redefined(lines, redefined)
    return compute_from_lines(lines)


def close_to(value: Optional[float], target: Optional[float]) -> bool:
    if value is None or target is None:
        return False
    return abs(value - target) < TOLERANCE


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #


def mutate_one_const(graph_lines: List[str], rng: random.Random) -> Optional[dict]:
    candidates = []
    for index, line in enumerate(graph_lines):
        match = CONST_LINE.match(line)
        if not match:
            continue
        try:
            candidates.append((index, match.group(1), float(match.group(2))))
        except ValueError:
            continue
    if not candidates:
        return None

    index, node_id, old_value = rng.choice(candidates)
    new_value = old_value + 1.0
    for _ in range(20):
        if abs(old_value) < 1e-9:
            candidate = float(rng.choice([-5, -3, -2, 2, 3, 5, 10]))
        else:
            candidate = old_value * rng.choice([0.5, 0.75, 1.25, 1.5, 2.0])
            if abs(candidate - old_value) < 1e-9:
                candidate += rng.choice([-3, -2, -1, 1, 2, 3])
        if abs(candidate - old_value) > 1e-9:
            new_value = candidate
            break

    if abs(new_value - round(new_value)) < 1e-9:
        new_value = float(int(round(new_value)))
    else:
        new_value = round(new_value, 4)

    mutated = list(graph_lines)
    new_line = f"{node_id}={new_value}"
    mutated[index] = new_line
    return {
        "mutation_type": "const",
        "mutated_graph": mutated,
        "line_index": index,
        "old_line": graph_lines[index],
        "new_line": new_line,
        "old_value": old_value,
        "new_value": new_value,
    }


def mutate_one_operation(graph_lines: List[str], rng: random.Random) -> Optional[dict]:
    candidates = []
    for index, line in enumerate(graph_lines):
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1].strip()
        if not re.search(r"[A-Za-z_]", rhs):
            continue
        ops = list(OP_PATTERN.finditer(rhs))
        if ops:
            candidates.append((index, rhs, ops))
    if not candidates:
        return None

    index, rhs, ops = rng.choice(candidates)
    op_match = rng.choice(ops)
    old_op = op_match.group(0)
    new_op = rng.choice([x for x in OP_SYMBOLS if x != old_op])
    new_rhs = rhs[: op_match.start()] + new_op + rhs[op_match.end() :]
    lhs = graph_lines[index].split("=", 1)[0].strip()
    mutated = list(graph_lines)
    new_line = f"{lhs}={new_rhs}"
    mutated[index] = new_line
    return {
        "mutation_type": "op",
        "mutated_graph": mutated,
        "line_index": index,
        "old_line": graph_lines[index],
        "new_line": new_line,
        "old_op": old_op,
        "new_op": new_op,
    }


def mutate_graph(
    graph_lines: List[str], rng: random.Random, const_prob: float = 0.5
) -> Optional[dict]:
    has_const = any(CONST_LINE.match(line) for line in graph_lines)
    has_op = any(
        "=" in line and OP_PATTERN.search(line.split("=", 1)[1]) for line in graph_lines
    )
    if not has_const and not has_op:
        return None
    if has_const and has_op:
        if rng.random() < const_prob:
            return mutate_one_const(graph_lines, rng) or mutate_one_operation(
                graph_lines, rng
            )
        return mutate_one_operation(graph_lines, rng) or mutate_one_const(
            graph_lines, rng
        )
    if has_const:
        return mutate_one_const(graph_lines, rng)
    return mutate_one_operation(graph_lines, rng)


def perturb_until_incorrect(
    graph_lines: List[str],
    item: dict,
    dataset_name: str,
    gold_value: float,
    rng: random.Random,
    const_prob: float,
    max_attempts: int = 30,
) -> Tuple[Optional[List[str]], Optional[dict]]:
    """CNF: mutate until executing the graph no longer reproduces the gold answer."""
    for _ in range(max_attempts):
        mutation = mutate_graph(list(graph_lines), rng, const_prob)
        if mutation is None:
            continue
        mutated = mutation["mutated_graph"]
        value = exec_lines(mutated, item, dataset_name)
        if value is not None and not close_to(value, gold_value):
            return mutated, mutation
    return None, None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def calc_f_id(sample: dict, dataset_name: str) -> Optional[bool]:
    pred = sample.get("pred_answer")
    if pred is None and sample.get("response"):
        pred = extract_answer_tag_number(sample["response"])
    if pred is None:
        return None
    graph = sample.get("generated_graph")
    if graph is None and sample.get("response"):
        graph = parse_graph_from_response(sample["response"])
    if graph is not None:
        graph_value = exec_lines(normalize_to_lines(graph), sample, dataset_name)
    else:
        # Runs that only stored the executed value rather than the graph itself.
        graph_value = sample.get("graph_value")
    if graph_value is None:
        return False
    return close_to(float(pred), float(graph_value))


def calc_f_ood(sample: dict, dataset_name: str) -> Optional[bool]:
    pred = sample.get("intervention_answer")
    if pred is None:
        return None
    graph = sample.get("intervention_graph") or sample.get("mutated_graph")
    if graph is None:
        return None
    expected = sample.get("expected_intervention_answer")
    if expected is None:
        expected = exec_lines(normalize_to_lines(graph), sample, dataset_name)
    if expected is None:
        return False
    try:
        return close_to(float(pred), float(expected))
    except Exception:
        return False


def aggregate_metrics(
    f_id: List[Optional[bool]],
    f_ood: List[Optional[bool]],
    f_strong: List[Optional[bool]],
    intervention_types: Optional[List[Optional[str]]] = None,
) -> Dict[str, Any]:
    def rate(values: List[Optional[bool]]) -> float:
        valid = [v for v in values if v is not None]
        if not valid:
            return float("nan")
        return sum(1 for v in valid if v) / len(valid)

    f_id_rate = rate(f_id)
    f_ood_rate = rate(f_ood)
    f_strong_rate = rate(f_strong)

    # Same denominator as F_strong, so the two numbers are directly comparable.
    aligned = [o for o, s in zip(f_ood, f_strong) if s is not None]
    f_ood_aligned = rate(aligned)

    metrics: Dict[str, Any] = {
        "n_samples": len(f_id),
        "n_f_id_valid": sum(1 for v in f_id if v is not None),
        "n_f_ood_valid": sum(1 for v in f_ood if v is not None),
        "n_f_strong_valid": sum(1 for v in f_strong if v is not None),
        "F_ID": f_id_rate,
        "F_OOD": f_ood_rate,
        "F_OOD_aligned": f_ood_aligned,
        "F_strong": f_strong_rate,
        "delta": (
            f_id_rate - f_ood_rate
            if math.isfinite(f_id_rate) and math.isfinite(f_ood_rate)
            else float("nan")
        ),
    }

    if intervention_types and len(intervention_types) == len(f_ood):
        for tag in ("COR", "CNF"):
            values = [
                v
                for v, t in zip(f_ood, intervention_types)
                if t == tag and v is not None
            ]
            metrics[f"n_{tag}"] = sum(1 for t in intervention_types if t == tag)
            metrics[f"n_{tag}_valid"] = len(values)
            metrics[f"F_OOD_{tag}"] = (
                sum(1 for v in values if v) / len(values) if values else float("nan")
            )
    return metrics


# --------------------------------------------------------------------------- #
# Offline mode
# --------------------------------------------------------------------------- #


def offline_faithfulness(args) -> dict:
    data = json.loads(Path(args.inference_json).read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    dataset_name = (
        data.get("dataset_name")
        or summary.get("dataset_name")
        or infer_dataset_name(
            data.get("test_file") or summary.get("test_file", ""), args.dataset_name
        )
    )
    rows = data.get("results") or data.get("samples") or []

    per_sample = []
    for row in rows:
        sample = dict(row)
        f_id = calc_f_id(sample, dataset_name)
        f_ood = calc_f_ood(sample, dataset_name)
        f_strong = None
        if f_id is not None and f_ood is not None:
            f_strong = bool(f_id and f_ood)
        per_sample.append(
            {
                "index": sample.get("index"),
                "question": (sample.get("question") or "")[:200],
                "F_ID": f_id,
                "F_OOD": f_ood,
                "F_strong": f_strong,
                "intervention_type": sample.get("intervention_type"),
            }
        )

    metrics = aggregate_metrics(
        [p["F_ID"] for p in per_sample],
        [p["F_OOD"] for p in per_sample],
        [p["F_strong"] for p in per_sample],
        [p["intervention_type"] for p in per_sample],
    )
    metrics.update(
        {
            "mode": "offline",
            "dataset_name": dataset_name,
            "source": str(args.inference_json),
        }
    )
    return {"metrics": metrics, "results": per_sample}


# --------------------------------------------------------------------------- #
# Online mode
# --------------------------------------------------------------------------- #


def is_gemma_model(path_like: str) -> bool:
    value = path_like.lower()
    return "gemma-2" in value or "gemma2" in value or "gemma_2" in value


def resolve_dtype(model_path: str, dtype: Optional[str]) -> str:
    requested = (dtype or "bfloat16").lower()
    if requested in ("auto", "default"):
        requested = "bfloat16"
    if is_gemma_model(model_path) and requested in ("float16", "fp16", "half"):
        print(f"Note: {model_path} -> bfloat16 (gemma2 rejects float16)", flush=True)
        return "bfloat16"
    return requested


def select_system_prompt(args, dataset_name: str) -> str:
    """Built-in generic prompt, unless an exact paper prompt is supplied."""
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8")
    if args.prompts_json:
        data = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        examples = data.get("examples", {})
        # The ICL runs used the dataset-specific few-shot exemplar prompts.
        if args.has_answer and dataset_name in examples:
            return examples[dataset_name]
        key = "SYSTEM_PROMPT_W_ANSWER" if args.has_answer else "SYSTEM_PROMPT"
        return data["prompts"][key]
    return SYSTEM_PROMPT_W_ANSWER if args.has_answer else SYSTEM_PROMPT_WO_ANSWER


def build_question_prompt(tokenizer, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{system_prompt}\n\nQuestion:\n{question}\n"


def build_answer_continuation_prompt(
    tokenizer, question: str, graph_lines: List[str], system_prompt: str
) -> str:
    """Force the model to answer *this* graph by prefilling it as its own reply."""
    prefix = json.dumps(graph_lines, ensure_ascii=False, indent=2) + "\nANSWER:"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefix},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    except Exception:
        pass
    return f"{system_prompt}\n\nQuestion:\n{question}\n\n{prefix}"


def generate_in_batches(llm, prompts, sampling, generation_kwargs, batch_size, label):
    outputs = []
    size = max(1, int(batch_size))
    total = (len(prompts) + size - 1) // size
    for start in range(0, len(prompts), size):
        print(f"[{label}] batch {start // size + 1}/{total}", flush=True)
        outputs.extend(llm.generate(prompts[start : start + size], sampling, **generation_kwargs))
    return outputs


def online_faithfulness(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    dataset_name = infer_dataset_name(args.test_file, args.dataset_name)
    system_prompt = select_system_prompt(args, dataset_name)

    rng = random.Random(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_data = load_split(args.test_file, args.split)
    if args.num_samples > 0:
        indices = list(range(len(test_data)))
        rng.shuffle(indices)
        sample_indices = sorted(indices[: min(args.num_samples, len(indices))])
    else:
        sample_indices = list(range(len(test_data)))

    use_lora = bool(args.checkpoint_path)
    dtype = resolve_dtype(args.model_base_path, args.dtype)
    tokenizer_path = args.model_base_path
    if use_lora and (Path(args.checkpoint_path) / "tokenizer_config.json").exists():
        tokenizer_path = args.checkpoint_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    if not getattr(tokenizer, "chat_template", None):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_base_path, trust_remote_code=args.trust_remote_code
        )
    if is_gemma_model(args.model_base_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    llm_kwargs = dict(
        model=args.model_base_path,
        tokenizer=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        enable_lora=use_lora,
        max_lora_rank=args.max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=dtype,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    generation_kwargs = {"use_tqdm": True}
    if use_lora:
        generation_kwargs["lora_request"] = LoRARequest("faithfulness", 1, args.checkpoint_path)

    graph_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed
    )
    answer_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.answer_max_new_tokens, seed=args.seed
    )

    prepared = []
    for index in sample_indices:
        item = test_data[index]
        question = get_question_text(item)
        if question:
            prepared.append({"index": index, "question": question, "item": item})

    graph_outputs = generate_in_batches(
        llm,
        [build_question_prompt(tokenizer, r["question"], system_prompt) for r in prepared],
        graph_sampling,
        generation_kwargs,
        args.batch_size,
        "graph",
    )

    per_sample: List[dict] = []
    interventions: List[dict] = []

    for row, output in zip(prepared, graph_outputs):
        response = output.outputs[0].text
        item = row["item"]
        gold_value = gold_answer_value(item, dataset_name)

        graph_obj = parse_graph_from_response(response)
        graph_lines = normalize_to_lines(graph_obj) if graph_obj else None
        graph_value = exec_lines(graph_lines, item, dataset_name)
        pred_answer = to_json_safe_real(extract_answer_tag_number(response))

        f_id: Optional[bool] = None
        if pred_answer is not None:
            f_id = close_to(pred_answer, graph_value) if graph_value is not None else False

        init_graph_correct = close_to(graph_value, gold_value)

        intervention_type: Optional[str] = None
        intervention_graph: Optional[List[str]] = None
        mutation_info: Optional[dict] = None

        if gold_value is not None:
            if init_graph_correct and graph_lines:
                intervention_type = "CNF"
                intervention_graph, mutation_info = perturb_until_incorrect(
                    graph_lines,
                    item,
                    dataset_name,
                    gold_value,
                    rng,
                    args.const_mutation_prob,
                )
            else:
                intervention_type = "COR"
                intervention_graph = gold_graph_lines(item)
                if intervention_graph is not None:
                    gold_exec = exec_lines(intervention_graph, item, dataset_name)
                    if not close_to(gold_exec, gold_value):
                        intervention_graph = None

        if intervention_type and intervention_graph:
            expected = exec_lines(intervention_graph, item, dataset_name)
            if expected is not None:
                interventions.append(
                    {
                        "index": row["index"],
                        "prompt": build_answer_continuation_prompt(
                            tokenizer, row["question"], intervention_graph, system_prompt
                        ),
                        "expected": expected,
                        "intervention_type": intervention_type,
                        "f_id": f_id,
                    }
                )

        per_sample.append(
            {
                "index": row["index"],
                "question": row["question"],
                "F_ID": f_id,
                "F_OOD": None,
                "F_strong": None,
                "pred_answer": pred_answer,
                "answer_parsed": pred_answer is not None,
                "answer_correct": close_to(pred_answer, gold_value),
                "graph_value": to_json_safe_real(graph_value),
                "gold_answer": to_json_safe_real(gold_value),
                "init_graph_correct": init_graph_correct,
                "intervention_type": intervention_type,
                "intervention_graph": intervention_graph,
                "mutation": mutation_info,
            }
        )

    if interventions:
        by_index = {p["index"]: p for p in per_sample}
        answer_outputs = generate_in_batches(
            llm,
            [x["prompt"] for x in interventions],
            answer_sampling,
            generation_kwargs,
            args.batch_size,
            "intervention",
        )
        for meta, output in zip(interventions, answer_outputs):
            continuation = output.outputs[0].text
            answer = to_json_safe_real(
                extract_answer_tag_number("ANSWER:" + continuation)
                or extract_answer_tag_number(continuation)
            )
            f_ood: Optional[bool] = None
            if answer is not None:
                f_ood = close_to(answer, meta["expected"])
            f_strong = (
                bool(meta["f_id"] and f_ood) if meta["f_id"] is not None and f_ood is not None else None
            )
            sample = by_index[meta["index"]]
            sample["F_OOD"] = f_ood
            sample["F_strong"] = f_strong
            sample["intervention_answer"] = answer
            sample["expected_intervention_answer"] = to_json_safe_real(meta["expected"])

    total = max(len(per_sample), 1)
    metrics = aggregate_metrics(
        [p["F_ID"] for p in per_sample],
        [p["F_OOD"] for p in per_sample],
        [p["F_strong"] for p in per_sample],
        [p["intervention_type"] for p in per_sample],
    )
    metrics.update(
        {
            "mode": "online",
            "dataset_name": dataset_name,
            "test_file": args.test_file,
            "split": args.split,
            "checkpoint_path": args.checkpoint_path or None,
            "num_samples_requested": args.num_samples,
            "n_evaluated": len(per_sample),
            "n_interventions": len(interventions),
            "has_answer": args.has_answer,
            "dtype": dtype,
            "ood_protocol": "COR_if_incorrect_else_CNF",
            "graph_execution_accuracy": sum(
                bool(p["init_graph_correct"]) for p in per_sample
            )
            / total,
            "answer_parse_rate": sum(bool(p["answer_parsed"]) for p in per_sample) / total,
            "answer_parsing_accuracy": sum(
                bool(p["answer_correct"]) for p in per_sample
            )
            / total,
        }
    )
    return {"metrics": metrics, "results": per_sample}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faithfulness eval (F_ID / F_OOD / F_strong) for computational graphs."
    )
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--inference-json", help="Offline: existing inference JSON")
    parser.add_argument("--model-base-path", help="Online: HF model id or path")
    parser.add_argument("--checkpoint-path", default="", help="Online: LoRA adapter ('' = base model / ICL)")
    parser.add_argument("--test-file", help="Online: dataset JSON")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-name",
        default=None,
        choices=["gsm8k", "gsm8k_new_ops", "igsm", "math", "math_new_ops"],
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Override the built-in graph system prompt with a text file",
    )
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="prompts/graph_inference_prompts.json, to reproduce the paper prompts exactly",
    )
    parser.add_argument("--has-answer", action="store_true")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all test items")
    parser.add_argument("--const-mutation-prob", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results_faithfulness")
    parser.add_argument("--output-stem", default="faithfulness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--answer-max-new-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.output_stem}.json"

    if args.mode == "offline":
        if not args.inference_json:
            raise SystemExit("offline mode requires --inference-json")
        payload = offline_faithfulness(args)
    else:
        if not args.model_base_path or not args.test_file:
            raise SystemExit("online mode requires --model-base-path and --test-file")
        payload = online_faithfulness(args)

    out_json.write_text(
        json.dumps(json_sanitize(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_sanitize(payload["metrics"]), indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
