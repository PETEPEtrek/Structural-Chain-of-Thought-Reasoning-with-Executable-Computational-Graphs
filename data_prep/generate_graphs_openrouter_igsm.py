#!/usr/bin/env python3
"""Generate IGSM computational graphs via OpenRouter."""

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
        "question": "The number of each Carrefour's Coffee equals 13. The number of each Safeway's Coffee equals 3. The number of each Coffee's Grapefruit equals the sum of each Almond Milk's Pomegranate, each Carrefour's Coffee and each Safeway's Coffee. The number of each Residential College District's Safeway equals 11. The number of each Residential College District's Carrefour equals 5 times as much as the sum of each Almond Milk's Grapefruit, each Coffee's Grapefruit and each Residential College District's Safeway. The number of each Carrefour's Almond Milk equals each Almond Milk's Ingredient. The number of each Almond Milk's Pomegranate equals each Safeway's Coffee. The number of each Safeway's Almond Milk equals each Almond Milk's Pomegranate. The number of each University District's Carrefour equals 22 times as much as each Carrefour's Coffee. The number of each Coffee's Pomegranate equals 15 times as much as each University District's Carrefour. The number of each Almond Milk's Grapefruit equals each University District's Carrefour. The number of each University District's Safeway equals 20 more than the difference of each Carrefour's Coffee and each University District's Carrefour. How many Product does Safeway have?",
        "solution": "Define Safeway's Coffee as B; so B = 3. Define Almond Milk's Pomegranate as f; so f = B = 3. Define Safeway's Almond Milk as x; so x = f = 3. Define Safeway's Product as v; so v = x + B = 3 + 3 = 6.",
        "graph": [
            {"id": "safeway_coffee", "type": "const", "value": 3},
            {"id": "safeway_almond_milk", "type": "const", "value": 3},
            {
                "id": "safeway_product",
                "type": "op",
                "op": "add",
                "inputs": ["safeway_almond_milk", "safeway_coffee"],
            },
        ],
    },
    {
        "question": "The number of each Hippopotamus Tank's Wolf equals 13. The number of each Wolf's Carpal Bones equals 4. The number of each Hippopotamus Tank's Camel equals 4 more than each Hippopotamus Tank's Wolf. The number of each Camel's Distal Phalanges equals 2 more than each Lemur Island's Animal. How many Carpal Bones does Wolf have?",
        "solution": "Define Wolf's Carpal Bones as b; so b = 4.",
        "graph": [
            {"id": "wolf_carpal_bones", "type": "const", "value": 4},
            {"id": "zero", "type": "const", "value": 0},
            {
                "id": "wolf_carpal_bones_result",
                "type": "op",
                "op": "add",
                "inputs": ["wolf_carpal_bones", "zero"],
            },
        ],
    },
]

SYSTEM_PROMPT = """You are an expert in computational graphs and mathematical reasoning.

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
The number of each Carrefour's Coffee equals 13. The number of each Safeway's Coffee equals 3. The number of each Coffee's Grapefruit equals the sum of each Almond Milk's Pomegranate, each Carrefour's Coffee and each Safeway's Coffee. The number of each Residential College District's Safeway equals 11. The number of each Residential College District's Carrefour equals 5 times as much as the sum of each Almond Milk's Grapefruit, each Coffee's Grapefruit and each Residential College District's Safeway. The number of each Carrefour's Almond Milk equals each Almond Milk's Ingredient. The number of each Almond Milk's Pomegranate equals each Safeway's Coffee. The number of each Safeway's Almond Milk equals each Almond Milk's Pomegranate. The number of each University District's Carrefour equals 22 times as much as each Carrefour's Coffee. The number of each Coffee's Pomegranate equals 15 times as much as each University District's Carrefour. The number of each Almond Milk's Grapefruit equals each University District's Carrefour. The number of each University District's Safeway equals 20 more than the difference of each Carrefour's Coffee and each University District's Carrefour. How many Product does Safeway have?

SOLUTION:
Define Safeway's Coffee as B; so B = 3. Define Almond Milk's Pomegranate as f; so f = B = 3. Define Safeway's Almond Milk as x; so x = f = 3. Define Safeway's Product as v; so v = x + B = 3 + 3 = 6.

COMPUTATIONAL GRAPH:
[
  {"id": "safeway_coffee", "type": "const", "value": 3},
  {"id": "safeway_almond_milk", "type": "const", "value": 3},
  {"id": "safeway_product", "type": "op", "op": "add", "inputs": ["safeway_almond_milk", "safeway_coffee"]}
]

Example 2
QUESTION:
The number of each Hippopotamus Tank's Wolf equals 13. The number of each Wolf's Carpal Bones equals 4. The number of each Hippopotamus Tank's Camel equals 4 more than each Hippopotamus Tank's Wolf. The number of each Camel's Distal Phalanges equals 2 more than each Lemur Island's Animal. How many Carpal Bones does Wolf have?

SOLUTION:
Define Wolf's Carpal Bones as b; so b = 4.

COMPUTATIONAL GRAPH:
[
  {"id": "wolf_carpal_bones", "type": "const", "value": 4},
  {"id": "zero", "type": "const", "value": 0},
  {"id": "wolf_carpal_bones_result", "type": "op", "op": "add", "inputs": ["wolf_carpal_bones", "zero"]}
]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IGSM computational graphs via OpenRouter."
    )
    parser.add_argument("--input", default="igsm.json")
    parser.add_argument("--output", default="igsm_graphs_generated_openrouter.json")
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
    return item.get("problem", "")


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
            "dataset": "igsm",
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

    for idx in tqdm(range(start_idx, len(input_data)), desc="Generating IGSM graphs"):
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
