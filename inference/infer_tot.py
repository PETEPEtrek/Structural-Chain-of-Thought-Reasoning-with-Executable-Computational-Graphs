#!/usr/bin/env python3
"""
Tree-of-Thoughts vLLM inference for GSM8K / GSM8K-NewOps / iGSM / MATH / EntailmentBank.

This is a base-model beam-search Tree-of-Thoughts baseline:
1. Keep a beam of partial reasoning states.
2. Expand each state with K next thoughts.
3. Ask the model to evaluate each partial state.
4. Keep the top beam_size states and repeat for max_depth.
5. Finish each final state and ask the model to choose the final answer/proof.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

DATASET_FILE_MAP = {
    "gsm8k": "gsm8k.json",
    "gsm8k_new_ops": "gsm8k_new_ops.json",
    "igsm": "igsm.json",
    "math": "math.json",
    "entailment": "entailment_bank.json",
}

MATH_THOUGHT_SYSTEM = """You are doing Tree-of-Thoughts search for a math word problem.
Generate exactly one next reasoning step that advances the partial solution.
Do not finish with a final answer unless the solution is already complete."""

MATH_EVAL_SYSTEM = """You are evaluating a partial Tree-of-Thoughts math solution.
Score whether the partial reasoning is useful, correct, and likely to lead to the right answer.
Return exactly:
SCORE: <integer from 0 to 10>
BRIEF: <one short reason>"""

MATH_FINISH_SYSTEM = """You are finishing a Tree-of-Thoughts math solution.
Continue from the selected partial reasoning state and produce the final answer.
End with exactly one final numeric answer inside <SOLUTION>...</SOLUTION>."""

MATH_JUDGE_SYSTEM = """You are the Tree-of-Thoughts judge.
Compare the candidate reasoning paths, identify the most reliable one, and produce the final answer.
Return a short justification and then exactly one final numeric answer inside <SOLUTION>...</SOLUTION>."""

ENTAIL_THOUGHT_SYSTEM = """You are doing Tree-of-Thoughts search for an EntailmentBank proof task.
Generate exactly one next proof/reasoning step that advances the partial proof.
Use sentence ids such as sent1, sent3 and intermediate ids such as int1 when useful.
Do not finish with the complete proof unless the proof is already complete."""

ENTAIL_EVAL_SYSTEM = """You are evaluating a partial Tree-of-Thoughts entailment proof.
Score whether the partial proof is useful, valid, and likely to lead to the hypothesis.
Return exactly:
SCORE: <integer from 0 to 10>
BRIEF: <one short reason>"""

ENTAIL_FINISH_SYSTEM = """You are finishing a Tree-of-Thoughts EntailmentBank proof.
Continue from the selected partial proof and produce the final proof.
End with exactly one proof inside <SOLUTION>...</SOLUTION>.
Proof format example: sent1 & sent3 -> int1: intermediate conclusion; int1 & sent2 -> hypothesis;"""

ENTAIL_JUDGE_SYSTEM = """You are the Tree-of-Thoughts judge for an EntailmentBank proof task.
Compare the candidate proofs and produce the best final proof.
Return a short justification and then exactly one proof inside <SOLUTION>...</SOLUTION>.
Use proof steps separated by semicolons."""

# gemma-2-2b ships without a chat template in some snapshots.
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


def is_gemma_2_2b(path_like: str) -> bool:
    value = path_like.lower()
    return "gemma-2-2b" in value or "gemma_2_2b" in value


def maybe_set_gemma_chat_template(tokenizer, model_path: str) -> None:
    if is_gemma_2_2b(model_path) and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Templates that reject the system role: fold it into the first user turn.
        merged: List[Dict[str, str]] = []
        prefix = ""
        for message in messages:
            if message["role"] == "system":
                prefix = f"System instruction:\n{message['content']}\n\n"
                continue
            content = message["content"]
            if prefix:
                content = prefix + content
                prefix = ""
            merged.append({"role": message["role"], "content": content})
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def load_json_records(path: Path, split: Optional[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        key = split or "test"
        if key not in data:
            raise KeyError(f"Split '{key}' not in {path}; available: {sorted(data)}")
        return data[key]
    return data


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def extract_solution_block(text: str) -> str:
    match = re.search(
        r"<SOLUTION>(.*?)</SOLUTION>", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def extract_numeric_answer(text: str) -> Optional[float]:
    block = extract_solution_block(text) or text
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", block)
    if not numbers:
        return None
    return parse_float(numbers[-1])


def normalize_proof(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"\s*;\s*", "; ", value)
    value = re.sub(r"\s*->\s*", " -> ", value)
    value = re.sub(r"\s*&\s*", " & ", value)
    value = re.sub(r"\s*:\s*", ": ", value)
    if value.endswith(";"):
        value = value[:-1]
    return " ".join(value.split())


def to_json_safe_real(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return to_json_safe_real(value)


def collect_token_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "total": float(sum(values)),
        "avg": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def get_question(item: Dict[str, Any], dataset: str) -> str:
    if dataset == "entailment":
        parts = []
        context = (item.get("context") or "").strip()
        if context:
            parts.append(f"context: {context}")
        question = (item.get("question") or "").strip()
        if question:
            parts.append(f"question: {question}")
        return "\n".join(parts)
    return item.get("question") or item.get("problem") or ""


def get_gold_math_answer(item: Dict[str, Any]) -> Optional[float]:
    if item.get("answer") is not None:
        return parse_float(item["answer"])
    return parse_float(extract_hash_answer(item.get("solution", "") or ""))


def build_expand_prompt(
    tokenizer,
    dataset: str,
    question: str,
    state_text: str,
    depth: int,
    thought_idx: int,
    k: int,
) -> str:
    shown_state = state_text or "[empty]"
    if dataset == "entailment":
        system = ENTAIL_THOUGHT_SYSTEM
        user = (
            f"{question}"
            f"\n\nCurrent partial proof/state:\n{shown_state}"
            f"\n\nDepth {depth}, branch {thought_idx} of {k}"
            ": generate one next proof step."
        )
    else:
        system = MATH_THOUGHT_SYSTEM
        user = (
            f"Problem:\n{question}"
            f"\n\nCurrent partial reasoning/state:\n{shown_state}"
            f"\n\nDepth {depth}, branch {thought_idx} of {k}"
            ": generate one next reasoning step."
        )
    return apply_chat_template(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


def build_eval_prompt(tokenizer, dataset: str, question: str, state_text: str) -> str:
    if dataset == "entailment":
        system = ENTAIL_EVAL_SYSTEM
        user = f"{question}\n\nPartial proof/state to score:\n{state_text}"
    else:
        system = MATH_EVAL_SYSTEM
        user = f"Problem:\n{question}\n\nPartial reasoning/state to score:\n{state_text}"
    return apply_chat_template(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


def build_finish_prompt(tokenizer, dataset: str, question: str, state_text: str) -> str:
    if dataset == "entailment":
        system = ENTAIL_FINISH_SYSTEM
        user = (
            f"{question}"
            f"\n\nSelected partial proof/state:\n{state_text}"
            "\n\nFinish the proof."
        )
    else:
        system = MATH_FINISH_SYSTEM
        user = (
            f"Problem:\n{question}"
            f"\n\nSelected partial reasoning/state:\n{state_text}"
            "\n\nFinish the solution."
        )
    return apply_chat_template(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


def build_judge_prompt(
    tokenizer, dataset: str, question: str, candidates: List[str]
) -> str:
    rendered = "\n\n".join(
        f"Candidate {i + 1}:\n{text}" for i, text in enumerate(candidates)
    )
    if dataset == "entailment":
        system = ENTAIL_JUDGE_SYSTEM
        user = f"{question}\n\nCandidate proofs:\n{rendered}\n\nChoose the best proof."
    else:
        system = MATH_JUDGE_SYSTEM
        user = (
            f"Problem:\n{question}"
            f"\n\nCandidate solutions:\n{rendered}"
            "\n\nChoose the best final numeric answer."
        )
    return apply_chat_template(
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


def parse_score(text: str) -> float:
    match = re.search(r"SCORE\s*:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.I)
    if match:
        return max(0.0, min(10.0, float(match.group(1))))
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return max(0.0, min(10.0, float(numbers[0])))
    return 0.0


def empty_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "num_generations": 0,
    }


def add_output_usage(acc: Dict[str, int], out) -> None:
    prompt_tokens = len(out.prompt_token_ids or [])
    completion_tokens = sum(len(c.token_ids or []) for c in out.outputs)
    acc["prompt_tokens"] += prompt_tokens
    acc["completion_tokens"] += completion_tokens
    acc["total_tokens"] += prompt_tokens + completion_tokens
    acc["num_generations"] += 1


def chunks(values: List[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def generate_batched(llm, prompts: List[str], sampling_params, batch_size: int):
    outputs = []
    for batch in chunks(prompts, max(1, batch_size)):
        outputs.extend(llm.generate(batch, sampling_params, use_tqdm=False))
    return outputs


def evaluate_entailment_prediction(
    model_proof: str, rec: Dict[str, Any]
) -> Dict[str, Any]:
    pred_norm = normalize_proof(model_proof)
    gold_norm = normalize_proof(rec.get("proof", "") or "")
    exact = pred_norm == gold_norm

    def step_set(value: str) -> set:
        return {x.strip() for x in value.split(";") if x.strip()}

    def token_set(value: str) -> set:
        return set(re.findall(r"[a-z]+\d*|hypothesis|int\d+", value.lower()))

    def f1(pred: set, gold: set) -> float:
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        common = len(pred & gold)
        if common == 0:
            return 0.0
        precision = common / len(pred)
        recall = common / len(gold)
        return 2 * precision * recall / (precision + recall)

    steps_f1 = f1(step_set(pred_norm), step_set(gold_norm))
    leaves_f1 = f1(
        {x for x in token_set(pred_norm) if x.startswith("sent")},
        {x for x in token_set(gold_norm) if x.startswith("sent")},
    )
    interm_f1 = f1(
        {
            x
            for x in token_set(pred_norm)
            if x.startswith("int") or x == "hypothesis"
        },
        {
            x
            for x in token_set(gold_norm)
            if x.startswith("int") or x == "hypothesis"
        },
    )
    return {
        "exact_normalized_match": float(exact),
        "proof-leaves": {"F1": leaves_f1, "AllCorrect": float(leaves_f1 == 1.0)},
        "proof-steps": {"F1": steps_f1, "AllCorrect": float(steps_f1 == 1.0)},
        "proof-intermediates": {"F1": interm_f1, "AllCorrect": float(interm_f1 == 1.0)},
        "proof-overall": {"AllCorrect": float(exact)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beam-search Tree-of-Thoughts vLLM inference."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm8k", "gsm8k_new_ops", "igsm", "math", "entailment"],
    )
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results_vllm_tot")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 = full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--candidate-temperature", type=float, default=0.7)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--judge-max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("Install: pip install vllm transformers tqdm")

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = args.dataset
    test_file = Path(args.test_file or DATASET_FILE_MAP[dataset])
    records = load_json_records(test_file, args.split)
    if args.limit:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base_path, trust_remote_code=args.trust_remote_code
    )
    maybe_set_gemma_chat_template(tokenizer, args.model_base_path)

    llm = LLM(
        model=args.model_base_path,
        tokenizer=args.model_base_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )

    expand_sampling = SamplingParams(
        temperature=args.candidate_temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    eval_sampling = SamplingParams(
        temperature=args.eval_temperature,
        top_p=1.0,
        max_tokens=args.eval_max_new_tokens,
    )
    finish_sampling = SamplingParams(
        temperature=args.eval_temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    judge_sampling = SamplingParams(
        temperature=args.judge_temperature,
        top_p=1.0,
        max_tokens=args.judge_max_new_tokens,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_tag = (
        Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    )
    out_name = (
        f"inference_vllm_tot_{dataset}_{model_tag}"
        f"_d{args.max_depth}_k{args.branching_factor}_b{args.beam_size}"
        f"_seed{args.seed}{args.results_suffix}.json"
    )
    out_path = results_dir / out_name

    examples = []
    for index, item in enumerate(records):
        question = get_question(item, dataset)
        if not question:
            continue
        if dataset == "entailment":
            examples.append(
                {
                    "index": index,
                    "item": item,
                    "question": question,
                    "gold_answer": item.get("answer"),
                    "gold_proof": item.get("proof", "") or "",
                }
            )
        else:
            gold_value = get_gold_math_answer(item)
            if gold_value is None:
                continue
            examples.append(
                {
                    "index": index,
                    "item": item,
                    "question": question,
                    "gold_value": gold_value,
                }
            )

    stage_usage = {stage: empty_usage() for stage in ("expand", "evaluate", "finish", "judge")}
    per_example_usage = [
        {**empty_usage(), "by_stage": {s: empty_usage() for s in stage_usage}}
        for _ in examples
    ]

    def record_usage(stage: str, example_idx: int, out) -> None:
        add_output_usage(stage_usage[stage], out)
        add_output_usage(per_example_usage[example_idx], out)
        add_output_usage(per_example_usage[example_idx]["by_stage"][stage], out)

    # One root state per example; beams evolve in lockstep so every stage is batched.
    beams: List[List[Dict[str, Any]]] = [
        [{"text": "", "score": 0.0, "depth": 0, "eval_response": ""}] for _ in examples
    ]
    traces: List[List[Dict[str, Any]]] = [[] for _ in examples]

    progress = tqdm(total=args.max_depth, desc=f"Beam ToT {dataset}")
    for depth in range(1, args.max_depth + 1):
        expand_jobs = []
        expand_prompts = []
        for ex_i, example in enumerate(examples):
            for state_i, state in enumerate(beams[ex_i]):
                for branch_i in range(1, args.branching_factor + 1):
                    expand_jobs.append(
                        {
                            "ex_i": ex_i,
                            "state_i": state_i,
                            "parent": state,
                            "branch_i": branch_i,
                        }
                    )
                    expand_prompts.append(
                        build_expand_prompt(
                            tokenizer,
                            dataset,
                            example["question"],
                            state["text"],
                            depth,
                            branch_i,
                            args.branching_factor,
                        )
                    )

        expand_outputs = generate_batched(
            llm, expand_prompts, expand_sampling, args.batch_size
        )

        candidates: List[List[Dict[str, Any]]] = [[] for _ in examples]
        for job, out in zip(expand_jobs, expand_outputs):
            record_usage("expand", job["ex_i"], out)
            thought = out.outputs[0].text.strip()
            parent = job["parent"]
            text = f"{parent['text']}\n{thought}".strip() if parent["text"] else thought
            candidates[job["ex_i"]].append(
                {
                    "text": text,
                    "score": 0.0,
                    "depth": depth,
                    "thought": thought,
                    "parent_score": parent["score"],
                    "eval_response": "",
                }
            )

        eval_jobs = []
        eval_prompts = []
        for ex_i, example in enumerate(examples):
            for cand_i, candidate in enumerate(candidates[ex_i]):
                eval_jobs.append({"ex_i": ex_i, "cand_i": cand_i})
                eval_prompts.append(
                    build_eval_prompt(
                        tokenizer, dataset, example["question"], candidate["text"]
                    )
                )

        eval_outputs = generate_batched(llm, eval_prompts, eval_sampling, args.batch_size)
        for job, out in zip(eval_jobs, eval_outputs):
            record_usage("evaluate", job["ex_i"], out)
            response = out.outputs[0].text
            candidate = candidates[job["ex_i"]][job["cand_i"]]
            candidate["eval_response"] = response
            candidate["score"] = parse_score(response)

        for ex_i in range(len(examples)):
            ranked = sorted(
                candidates[ex_i], key=lambda c: c["score"], reverse=True
            )[: args.beam_size]
            beams[ex_i] = ranked
            traces[ex_i].append(
                {
                    "depth": depth,
                    "num_candidates": len(candidates[ex_i]),
                    "selected": [
                        {
                            "score": c["score"],
                            "text": c["text"],
                            "eval_response": c["eval_response"],
                        }
                        for c in ranked
                    ],
                }
            )
        progress.update(1)
    progress.close()

    finish_jobs = []
    finish_prompts = []
    for ex_i, example in enumerate(examples):
        for state_i, state in enumerate(beams[ex_i]):
            finish_jobs.append({"ex_i": ex_i, "state_i": state_i, "state": state})
            finish_prompts.append(
                build_finish_prompt(
                    tokenizer, dataset, example["question"], state["text"]
                )
            )

    finish_outputs = generate_batched(
        llm, finish_prompts, finish_sampling, args.batch_size
    )
    finals: List[List[Dict[str, Any]]] = [[] for _ in examples]
    for job, out in zip(finish_jobs, finish_outputs):
        record_usage("finish", job["ex_i"], out)
        finals[job["ex_i"]].append(
            {
                "state_score": job["state"]["score"],
                "state_text": job["state"]["text"],
                "final_response": out.outputs[0].text,
            }
        )

    judge_prompts = [
        build_judge_prompt(
            tokenizer,
            dataset,
            example["question"],
            [c["final_response"] for c in finals[ex_i]],
        )
        for ex_i, example in enumerate(examples)
    ]
    judge_outputs = generate_batched(
        llm, judge_prompts, judge_sampling, args.batch_size
    )

    results = []
    correct = 0
    entail_totals = {
        "exact_normalized_match": 0.0,
        "overall_accuracy": 0.0,
        "leaves_f1": 0.0,
        "steps_f1": 0.0,
        "intermediates_f1": 0.0,
    }

    for ex_i, (example, out) in enumerate(zip(examples, judge_outputs)):
        record_usage("judge", ex_i, out)
        response = out.outputs[0].text
        item = example["item"]

        if dataset == "entailment":
            model_proof = extract_solution_block(response) or response.strip()
            metrics = evaluate_entailment_prediction(model_proof, item)
            is_correct = bool(metrics["proof-overall"]["AllCorrect"])
            entail_totals["exact_normalized_match"] += metrics["exact_normalized_match"]
            entail_totals["overall_accuracy"] += metrics["proof-overall"]["AllCorrect"]
            entail_totals["leaves_f1"] += metrics["proof-leaves"]["F1"]
            entail_totals["steps_f1"] += metrics["proof-steps"]["F1"]
            entail_totals["intermediates_f1"] += metrics["proof-intermediates"]["F1"]
            correct += int(is_correct)
            results.append(
                {
                    "index": example["index"],
                    "id": item.get("id"),
                    "question": example["question"],
                    "answer": example["gold_answer"],
                    "context": item.get("context"),
                    "ground_truth_proof": example["gold_proof"],
                    "tot_trace": traces[ex_i],
                    "final_candidates": finals[ex_i],
                    "model_response": response,
                    "model_proof": model_proof,
                    "metrics": metrics,
                    "is_correct": is_correct,
                    "token_usage": per_example_usage[ex_i],
                }
            )
        else:
            model_answer = extract_numeric_answer(response)
            gold_value = example["gold_value"]
            is_correct = (
                model_answer is not None and abs(model_answer - gold_value) < 1e-6
            )
            correct += int(is_correct)
            results.append(
                {
                    "index": example["index"],
                    "question": example["question"],
                    "solution": item.get("solution", ""),
                    "ground_truth_answer": gold_value,
                    "tot_trace": traces[ex_i],
                    "final_candidates": finals[ex_i],
                    "model_response": response,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "token_usage": per_example_usage[ex_i],
                }
            )

    total = len(results)
    accuracy = correct / total if total else 0.0
    summary = {
        "mode": "tree_of_thoughts_vllm",
        "dataset": dataset,
        "model_base_path": args.model_base_path,
        "test_file": str(test_file),
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "algorithm": "beam_search_tree_of_thoughts",
        "max_depth": args.max_depth,
        "branching_factor": args.branching_factor,
        "beam_size": args.beam_size,
        "batch_size": args.batch_size,
        "candidate_temperature": args.candidate_temperature,
        "eval_temperature": args.eval_temperature,
        "judge_temperature": args.judge_temperature,
        "seed": args.seed,
        "results_file": str(out_path),
        "token_usage": {
            "prompt": collect_token_stats(
                [u["prompt_tokens"] for u in per_example_usage]
            ),
            "completion": collect_token_stats(
                [u["completion_tokens"] for u in per_example_usage]
            ),
            "total": collect_token_stats(
                [u["total_tokens"] for u in per_example_usage]
            ),
            "by_stage": stage_usage,
        },
    }
    if dataset == "entailment" and total:
        summary["entailment_metrics"] = {
            key: value / total for key, value in entail_totals.items()
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            json_sanitize({"summary": summary, "results": results}),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_path}")
    print(f"Total={total} Correct={correct} Accuracy={100 * accuracy:.2f}%")
    if dataset == "entailment" and total:
        metrics = summary["entailment_metrics"]
        print(
            "Entailment: exact={:.2f}% overall={:.2f}% leaves_f1={:.2f}% steps_f1={:.2f}%".format(
                100 * metrics["exact_normalized_match"],
                100 * metrics["overall_accuracy"],
                100 * metrics["leaves_f1"],
                100 * metrics["steps_f1"],
            )
        )


if __name__ == "__main__":
    main()
