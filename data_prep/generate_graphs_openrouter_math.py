#!/usr/bin/env python3
"""Generate MATH computational graphs via OpenRouter."""

import argparse
import json
import math
import os
import time
from fractions import Fraction

from openai import OpenAI
from tqdm import tqdm

ALLOWED_OPS = {"add", "sub", "mul", "div", "pow", "mod"}

FEWSHOT_EXAMPLES = [
    {
        "question": "An octopus told me that his underwater cave is $245_{8}$ years old. How many years is this in base ten?",
        "solution": "$245_{8} = 5\\cdot8^{0}+4\\cdot8^{1}+2\\cdot8^{2} = 5+32+128= \\boxed{165}$.",
        "graph": [
            {"id": "digit_5", "type": "const", "value": 5},
            {"id": "digit_4", "type": "const", "value": 4},
            {"id": "digit_2", "type": "const", "value": 2},
            {"id": "base_8", "type": "const", "value": 8},
            {"id": "exp_0", "type": "const", "value": 0},
            {"id": "exp_1", "type": "const", "value": 1},
            {"id": "exp_2", "type": "const", "value": 2},
            {"id": "base_pow_0", "type": "op", "op": "pow", "inputs": ["base_8", "exp_0"]},
            {"id": "base_pow_1", "type": "op", "op": "pow", "inputs": ["base_8", "exp_1"]},
            {"id": "base_pow_2", "type": "op", "op": "pow", "inputs": ["base_8", "exp_2"]},
            {"id": "term_0", "type": "op", "op": "mul", "inputs": ["digit_5", "base_pow_0"]},
            {"id": "term_1", "type": "op", "op": "mul", "inputs": ["digit_4", "base_pow_1"]},
            {"id": "term_2", "type": "op", "op": "mul", "inputs": ["digit_2", "base_pow_2"]},
            {"id": "base10_value", "type": "op", "op": "add", "inputs": ["term_0", "term_1", "term_2"]},
        ],
    },
    {
        "question": "Find the value of $x,$ if \\[|x-20| + |x-18| = |2x-36|.\\]",
        "solution": "We recognize that $|2x-36| = 2|x-18|,$ so we get \\[|x-20| = |x-18|.\\]This means that, on the number line, $x$ is equidistant from $20$ and $18.$ Therefore $x$ must lie halfway between $20$ and $18,$ so \\[x = \\frac{20+18}{2} = \\boxed{19}.\\]",
        "graph": [
            {"id": "twenty", "type": "const", "value": 20},
            {"id": "eighteen", "type": "const", "value": 18},
            {"id": "two", "type": "const", "value": 2},
            {"id": "sum_points", "type": "op", "op": "add", "inputs": ["twenty", "eighteen"]},
            {"id": "x", "type": "op", "op": "div", "inputs": ["sum_points", "two"]},
        ],
    },
]

SYSTEM_PROMPT = r"""You are an expert in computational graphs and mathematical reasoning.

Given QUESTION and SOLUTION, generate only a computational graph as a JSON array.
Each node must be one of:
1) {"id": "<name>", "type": "const", "value": <number>}
2) {"id": "<name>", "type": "op", "op": "<operation>", "inputs": ["<id1>", "<id2>", ...]}

Rules:
1. Allowed operations: "add", "sub", "mul", "div", "pow", "mod"
2. Output only JSON array, no extra text.
3. List nodes in computational order.
4. Every op input must reference a previously defined node id.
5. The last node must be an operation node, not a constant.
6. Use only numbers and operations needed to solve the task.

Example 1
QUESTION:
An octopus told me that his underwater cave is $245_{8}$ years old. How many years is this in base ten?

SOLUTION:
$245_{8} = 5\cdot8^{0}+4\cdot8^{1}+2\cdot8^{2} = 5+32+128= \boxed{165}$.

COMPUTATIONAL GRAPH:
[
  {"id": "digit_5", "type": "const", "value": 5},
  {"id": "digit_4", "type": "const", "value": 4},
  {"id": "digit_2", "type": "const", "value": 2},
  {"id": "base_8", "type": "const", "value": 8},
  {"id": "exp_0", "type": "const", "value": 0},
  {"id": "exp_1", "type": "const", "value": 1},
  {"id": "exp_2", "type": "const", "value": 2},
  {"id": "base_pow_0", "type": "op", "op": "pow", "inputs": ["base_8", "exp_0"]},
  {"id": "base_pow_1", "type": "op", "op": "pow", "inputs": ["base_8", "exp_1"]},
  {"id": "base_pow_2", "type": "op", "op": "pow", "inputs": ["base_8", "exp_2"]},
  {"id": "term_0", "type": "op", "op": "mul", "inputs": ["digit_5", "base_pow_0"]},
  {"id": "term_1", "type": "op", "op": "mul", "inputs": ["digit_4", "base_pow_1"]},
  {"id": "term_2", "type": "op", "op": "mul", "inputs": ["digit_2", "base_pow_2"]},
  {"id": "base10_value", "type": "op", "op": "add", "inputs": ["term_0", "term_1", "term_2"]}
]

Example 2
QUESTION:
Find the value of $x,$ if \[|x-20| + |x-18| = |2x-36|.\]

SOLUTION:
We recognize that $|2x-36| = 2|x-18|,$ so we get \[|x-20| = |x-18|.\]This means that, on the number line, $x$ is equidistant from $20$ and $18.$ Therefore $x$ must lie halfway between $20$ and $18,$ so \[x = \frac{20+18}{2} = \boxed{19}.\]

COMPUTATIONAL GRAPH:
[
  {"id": "twenty", "type": "const", "value": 20},
  {"id": "eighteen", "type": "const", "value": 18},
  {"id": "two", "type": "const", "value": 2},
  {"id": "sum_points", "type": "op", "op": "add", "inputs": ["twenty", "eighteen"]},
  {"id": "x", "type": "op", "op": "div", "inputs": ["sum_points", "two"]}
]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MATH computational graphs via OpenRouter."
    )
    parser.add_argument("--input", default="math.json")
    parser.add_argument("--output", default="math_graphs_generated_openrouter.json")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--sleep-between", type=float, default=0.5)
    return parser.parse_args()


def parse_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip().replace(",", "")
    try:
        return float(value_str)
    except ValueError:
        pass
    if "/" in value_str:
        try:
            return float(Fraction(value_str))
        except Exception:
            pass
    raise ValueError("Cannot parse number")


def extract_json_array(text: str):
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except Exception:
                    return None
    return None


def get_question_text(item: dict) -> str:
    return item.get("question", "")


def get_solution_text(item: dict) -> str:
    return item.get("solution", "")


def get_target_answer(item: dict):
    answer = item.get("answer")
    if answer is None:
        return None
    try:
        return parse_float(answer)
    except Exception:
        return None


def apply_op(op_name: str, operands: list[float]) -> float:
    result = operands[0]
    for value in operands[1:]:
        if op_name == "add":
            result += value
        elif op_name == "sub":
            result -= value
        elif op_name == "mul":
            result *= value
        elif op_name == "div":
            result /= value
        elif op_name == "pow":
            result **= value
        elif op_name == "mod":
            result %= value
        else:
            raise ValueError("Unsupported operation")
    return result


def ensure_real_finite_number(value, context: str) -> float:
    if isinstance(value, complex):
        raise ValueError(f"{context} produced complex value")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} produced non-numeric value")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{context} produced non-finite value")
    return numeric_value


def resolve_input_value(input_ref, values: dict[str, float]):
    if isinstance(input_ref, str):
        if input_ref in values:
            return values[input_ref]
        return parse_float(input_ref)
    if isinstance(input_ref, (int, float)):
        return parse_float(input_ref)
    if isinstance(input_ref, dict):
        ref_id = input_ref.get("id")
        if isinstance(ref_id, str) and ref_id in values:
            return values[ref_id]
        if "value" in input_ref:
            return parse_float(input_ref["value"])
    raise ValueError(f"Unsupported input reference: {input_ref!r}")


def validate_graph(graph_obj, expected_answer: float):
    errors = []
    if not isinstance(graph_obj, list) or not graph_obj:
        return False, None, ["Graph is not a non-empty JSON array"]

    values = {}
    node_ids = set()

    for node in graph_obj:
        if not isinstance(node, dict):
            errors.append("Node is not an object")
            continue

        node_id = node.get("id")
        node_type = node.get("type")
        if not node_id or not isinstance(node_id, str):
            errors.append("Node id is missing or invalid")
            continue
        if node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)

        if node_type == "const":
            if "value" not in node:
                errors.append(f"Const node {node_id} has no value")
                continue
            try:
                values[node_id] = parse_float(node["value"])
            except Exception:
                errors.append(f"Const node {node_id} value is invalid")
        elif node_type == "op":
            op_name = node.get("op")
            inputs = node.get("inputs")
            if op_name not in ALLOWED_OPS:
                errors.append(f"Op node {node_id} has unsupported op")
                continue
            if not isinstance(inputs, list) or not inputs:
                errors.append(f"Op node {node_id} has invalid inputs")
                continue
            try:
                operands = [
                    resolve_input_value(input_ref=input_ref, values=values)
                    for input_ref in inputs
                ]
                raw_value = apply_op(op_name, operands)
                values[node_id] = ensure_real_finite_number(
                    raw_value, context=f"Op node {node_id}"
                )
            except Exception as exc:
                errors.append(f"Op eval failed for {node_id}: {exc}")
        else:
            errors.append(f"Node {node_id} has invalid type")

    if errors:
        return False, None, errors

    last_node = graph_obj[-1]
    if last_node.get("type") != "op":
        return False, None, ["Last node must be op, not const"]

    final_id = last_node.get("id")
    if final_id not in values:
        return False, None, ["Last node value was not computed"]
    try:
        final_value = ensure_real_finite_number(
            values[final_id], context=f"Last node {final_id}"
        )
    except Exception as exc:
        return False, None, [str(exc)]

    if expected_answer is None:
        return False, final_value, ["Ground truth answer is missing"]
    if abs(final_value - expected_answer) >= 1e-6:
        return False, final_value, ["Graph answer does not match target answer"]

    return True, final_value, []


def build_client(api_key: str):
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def build_user_prompt(question: str, solution: str) -> str:
    return f"QUESTION:\n{question}\n\nSOLUTION:\n{solution}\n"


def generate_graph_text(
    client,
    model: str,
    system_prompt: str,
    question: str,
    solution: str,
    max_retries: int,
):
    user_prompt = build_user_prompt(question, solution)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)
    return None


def save_payload(output_path: str, model: str, input_path: str, results: list[dict]) -> None:
    valid = sum(1 for item in results if item.get("is_valid"))
    payload = {
        "meta": {
            "dataset": "math",
            "input": input_path,
            "model": model,
            "total": len(results),
            "valid": valid,
            "valid_rate": (valid / len(results)) if results else 0.0,
            "fewshot_count": len(FEWSHOT_EXAMPLES),
            "fewshot_examples": FEWSHOT_EXAMPLES,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_existing_results(output_path: str, total_items: int) -> list[dict]:
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception:
        return []

    loaded = payload.get("results", [])
    if not isinstance(loaded, list):
        return []

    dedup: dict[int, dict] = {}
    for entry in loaded:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if isinstance(idx, int) and 0 <= idx < total_items:
            dedup[idx] = entry
    return [dedup[idx] for idx in sorted(dedup)]


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    with open(args.input, "r", encoding="utf-8") as file_obj:
        input_data = json.load(file_obj)
    if args.max_examples > 0:
        input_data = input_data[: args.max_examples]

    client = build_client(api_key)

    system_prompt = SYSTEM_PROMPT

    results = load_existing_results(output_path=args.output, total_items=len(input_data))
    processed_indices = {entry["index"] for entry in results if "index" in entry}
    valid_count = sum(1 for item in results if item.get("is_valid"))
    start_idx = next(
        (idx for idx in range(len(input_data)) if idx not in processed_indices),
        len(input_data),
    )

    for idx in tqdm(range(start_idx, len(input_data)), desc="Generating MATH graphs"):
        if idx in processed_indices:
            continue
        item = input_data[idx]
        question = get_question_text(item)
        solution = get_solution_text(item)
        answer = get_target_answer(item)

        generated = generate_graph_text(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            question=question,
            solution=solution,
            max_retries=args.max_retries,
        )

        if generated is None:
            entry = {
                "index": idx,
                "question": question,
                "solution": solution,
                "answer": answer,
                "raw_response": None,
                "computational_graph": None,
                "graph_answer": None,
                "is_valid": False,
                "errors": ["generation_failed"],
            }
        else:
            graph_obj = extract_json_array(generated)
            if graph_obj is None:
                entry = {
                    "index": idx,
                    "question": question,
                    "solution": solution,
                    "answer": answer,
                    "raw_response": generated,
                    "computational_graph": None,
                    "graph_answer": None,
                    "is_valid": False,
                    "errors": ["cannot_parse_json_array"],
                }
            else:
                is_valid, graph_answer, errors = validate_graph(graph_obj, answer)
                if is_valid:
                    valid_count += 1
                entry = {
                    "index": idx,
                    "question": question,
                    "solution": solution,
                    "answer": answer,
                    "raw_response": generated,
                    "computational_graph": graph_obj,
                    "graph_answer": graph_answer,
                    "is_valid": is_valid,
                    "errors": errors,
                }

        results.append(entry)
        processed_indices.add(idx)
        results.sort(key=lambda row: row.get("index", -1))
        save_payload(
            output_path=args.output,
            model=args.model,
            input_path=args.input,
            results=results,
        )
        time.sleep(args.sleep_between)

    print(f"Saved: {args.output}")
    print(f"Valid graphs: {valid_count}/{len(results)}")


if __name__ == "__main__":
    main()
