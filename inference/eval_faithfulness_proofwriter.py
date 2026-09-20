#!/usr/bin/env python3
"""
ProofWriter faithfulness (F_ID / F_OOD / F_strong) with vLLM.

Standalone: imports only pip packages, never another file in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Inlined from the original sibling modules so this file stands alone.
# --------------------------------------------------------------------------- #

THINK_START = "<think>"


THINK_END = "</think>"


SOLUTION_START = "<solution>"


SOLUTION_END = "</solution>"


GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'assistant' %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'model\n' }}{% endif %}"""


SYSTEM_PROMPT_ANSWER_SFT = """You solve closed-world ProofWriter entailment tasks.
Use only the supplied facts and rules. Return a compact proof expression, then the boolean:
PROOF: <proof-expression>
ANSWER: True
or
PROOF: <proof-expression>
ANSWER: False

The PROOF line uses indexed facts/rules only, e.g. triple7 or
(((triple4) -> rule3) -> rule6) -> rule5. No free-form reasoning."""


SYSTEM_PROMPT_COT_ANSWER_SFT = """You solve closed-world ProofWriter entailment tasks.
Reason step by step from the supplied facts and rules, without inventing facts.
Mention the indexed rule used at every inference step. After the reasoning, emit
PROOF: <proof-expression>
then end with exactly ANSWER: True or ANSWER: False.
The PROOF line uses indexed facts/rules only, e.g. triple7 or
(((triple4) -> rule3) -> rule6) -> rule5."""


SYSTEM_PROMPT_DEFAULT_GRPO = """You solve closed-world ProofWriter entailment tasks.
Use only the supplied facts and rules.
Think about the problem and provide step-by-step reasoning.
Place the reasoning between <think> and </think>.
Then provide the final answer between <solution> and </solution>.
Inside the thinking section, write genuine multi-step reasoning in the same style
as CoT+answer training: start from known facts, mention the indexed rule at every
inference step (for example: "Given Fiona is cold., apply rule3 (All cold things
are round.) to conclude Fiona is round."), and do not invent facts.
Inside the solution section, put exactly:
PROOF: <proof-expression>
ANSWER: True|False
The PROOF line uses indexed facts/rules only, e.g. triple7 or
(((triple4) -> rule3) -> rule6) -> rule5. Tags are lowercase."""


_GRAPH_INSTRUCTIONS = """You solve ProofWriter tasks by emitting a full proof graph.
Return one JSON array followed by exactly ANSWER: True or ANSWER: False.
Do not use XML tags, markdown, or any other text.
Use the same graph style as EntailmentBank:
- A source fact is {"id":"tripleN","type":"const","value":"..."}.
- A rule application is {"id":"intN","type":"op","op":"apply_rule",
  "rule":"ruleN","inputs":["tripleN","intM"],"value":"..."}.
- inputs are ordered and may reference source facts or earlier rule applications.
- The final array node proves the question for True, or its explicit complement
  for False. Name a derived final node "conclusion"; a direct fact may retain its
  tripleN id. Include only nodes participating in that proof.
- The ANSWER line must agree with the literal proved by the final graph node."""


_GRAPH_FEW_SHOTS = r"""

Few-shot example 1:
INPUT:
facts:
triple1: Bob is green.
rules:
rule1: If someone is green then they are quiet.
question: Bob is quiet.
OUTPUT:
[{"id":"triple1","type":"const","value":"Bob is green."},{"id":"conclusion","type":"op","op":"apply_rule","rule":"rule1","inputs":["triple1"],"value":"Bob is quiet."}]
ANSWER: True

Few-shot example 2:
INPUT:
facts:
triple1: Anne is red.
triple2: Anne is not young.
rules:
rule1: If someone is red then they are kind.
question: Anne is young.
OUTPUT:
[{"id":"triple2","type":"const","value":"Anne is not young."}]
ANSWER: False"""


SYSTEM_PROMPT_GRAPH_SFT = _GRAPH_INSTRUCTIONS + _GRAPH_FEW_SHOTS


SYSTEM_PROMPT_GRAPH_GRPO = _GRAPH_INSTRUCTIONS + _GRAPH_FEW_SHOTS


def is_gemma_2_2b(path_like: str) -> bool:
    value = str(path_like).lower()
    return "gemma-2-2b" in value or "gemma_2_2b" in value


def maybe_set_gemma_chat_template(tokenizer: Any, model_path: str) -> None:
    """Allow Gemma 2 to consume the system-role prompts used by ProofWriter."""
    if is_gemma_2_2b(model_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM


def normalize_boolean(value: Any) -> str:
    answer = str(value).strip().casefold()
    if answer == "true":
        return "True"
    if answer == "false":
        return "False"
    raise ValueError(f"ProofWriter answer must be True or False, got {value!r}")


def build_user_input(item: Mapping[str, Any]) -> str:
    facts = item.get("facts") or {}
    rules = item.get("rules") or {}
    fact_lines = [f"{key}: {value}" for key, value in facts.items()]
    rule_lines = [f"{key}: {value}" for key, value in rules.items()]
    return "\n".join(
        [
            "theory:",
            str(item.get("theory", "")).strip(),
            "facts:",
            *fact_lines,
            "rules:",
            *rule_lines,
            "question:",
            str(item.get("question", "")).strip(),
        ]
    )


def _reasoning_text(item: Mapping[str, Any]) -> str:
    reasoning = item.get("canonical_reasoning") or []
    if isinstance(reasoning, list):
        text = "\n".join(str(step).strip() for step in reasoning if str(step).strip())
    else:
        text = str(reasoning).strip()
    return text or "The indexed theory directly supports the selected boolean."


def proof_expression(item: Mapping[str, Any]) -> str:
    """Return the canonical ProofWriter proof string (triple/rule expression)."""
    proof = str(item.get("canonical_proof") or "").strip()
    if proof:
        return proof
    gold_proofs = item.get("gold_proofs") or []
    if isinstance(gold_proofs, list):
        for entry in gold_proofs:
            if isinstance(entry, dict):
                expression = str(entry.get("expression") or "").strip()
                if expression:
                    return expression
    raise ValueError(f"Row {item.get('id', '<unknown>')} has no canonical_proof")


def proof_answer_block(item: Mapping[str, Any]) -> str:
    """Shared trailing block used by answer SFT, CoT SFT, and default GRPO."""
    answer = normalize_boolean(item.get("answer"))
    return f"PROOF: {proof_expression(item)}\nANSWER: {answer}"


def answer_target(item: Mapping[str, Any]) -> str:
    return proof_answer_block(item)


def cot_answer_target(item: Mapping[str, Any]) -> str:
    return f"{_reasoning_text(item)}\n{proof_answer_block(item)}"


def default_grpo_full_target(item: Mapping[str, Any]) -> str:
    """Full gold Default-GRPO response: CoT reasoning in think + PROOF/ANSWER."""
    reasoning = _reasoning_text(item).strip()
    return (
        f"{THINK_START}\n{reasoning}\n{THINK_END}"
        f"{SOLUTION_START}\n{proof_answer_block(item)}\n{SOLUTION_END}"
    )


def default_grpo_solution_target(item: Mapping[str, Any]) -> str:
    """Gold solution body (PROOF/ANSWER) for format checks / demos."""
    return proof_answer_block(item)


def to_entailment_graph(graph: Any) -> list[dict[str, Any]]:
    """Convert the stored bipartite proof into EntailmentBank's JSON-array style."""
    if isinstance(graph, list):
        return graph
    if not isinstance(graph, dict) or set(graph) != {"nodes", "edges", "root"}:
        raise ValueError("Expected a stored nodes/edges/root proof graph")

    by_id = {node["id"]: node for node in graph["nodes"]}
    producer: dict[str, str] = {}
    premises: dict[str, list[tuple[int, str]]] = {}
    for edge in graph["edges"]:
        if edge["relation"] == "result":
            producer[edge["target"]] = edge["source"]
        elif edge["relation"] == "premise":
            premises.setdefault(edge["target"], []).append(
                (int(edge["order"]), edge["source"])
            )

    result_for_op = {
        edge["source"]: edge["target"]
        for edge in graph["edges"]
        if edge["relation"] == "result"
    }
    output: list[dict[str, Any]] = []
    emitted: dict[str, str] = {}
    intermediate_count = 0

    def emit_const(node_id: str, is_root: bool = False) -> str:
        nonlocal intermediate_count
        if node_id in emitted:
            return emitted[node_id]
        node = by_id[node_id]
        if node_id not in producer:
            fact_id = str(node.get("fact_id", "")).strip()
            if not fact_id:
                raise ValueError(f"Unsupported derived const node {node_id}")
            output.append(
                {"id": fact_id, "type": "const", "value": str(node["text"])}
            )
            emitted[node_id] = fact_id
            return fact_id

        op_id = producer[node_id]
        op = by_id[op_id]
        input_ids = [
            emit_const(source)
            for _, source in sorted(premises.get(op_id, []), key=lambda pair: pair[0])
        ]
        intermediate_count += 1
        compact_id = "conclusion" if is_root else f"int{intermediate_count}"
        output.append(
            {
                "id": compact_id,
                "type": "op",
                "op": "apply_rule",
                "rule": str(op["rule_id"]),
                "inputs": input_ids,
                "value": str(by_id[result_for_op[op_id]]["text"]),
            }
        )
        emitted[node_id] = compact_id
        return compact_id

    emit_const(str(graph["root"]), is_root=True)
    return output


def graph_target(item: Mapping[str, Any]) -> str:
    graph = to_entailment_graph(item.get("canonical_graph"))
    answer = normalize_boolean(item.get("answer"))
    rendered = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    return f"{rendered}\nANSWER: {answer}"


def get_system_prompt(method: str) -> str:
    prompts = {
        "answer_sft": SYSTEM_PROMPT_ANSWER_SFT,
        "cot_answer_sft": SYSTEM_PROMPT_COT_ANSWER_SFT,
        "default_grpo": SYSTEM_PROMPT_DEFAULT_GRPO,
        "graph_sft": SYSTEM_PROMPT_GRAPH_SFT,
        "graph_grpo": SYSTEM_PROMPT_GRAPH_GRPO,
    }
    try:
        return prompts[method]
    except KeyError as exc:
        raise ValueError(f"Unknown ProofWriter method: {method}") from exc


FULL_RESPONSE_RE = re.compile(
    r"\A\s*<think>(?P<think>.+?)</think><solution>\s*"
    r"PROOF:\s*(?P<proof>.+?)\s*\n\s*ANSWER:\s*(?P<answer>True|False)\s*"
    r"</solution>\s*\Z",
    flags=re.DOTALL,
)


def parse_tagged_response(text: Any) -> tuple[str, str] | None:
    """Return (think, answer) when the default-GRPO tagged format matches."""
    match = FULL_RESPONSE_RE.fullmatch(str(text or ""))
    if not match or not match.group("think").strip() or not match.group("proof").strip():
        return None
    return match.group("think").strip(), match.group("answer")


def parse_tagged_response_with_proof(
    text: Any,
) -> tuple[str, str, str] | None:
    """Return (think, proof, answer) for default-GRPO outputs."""
    match = FULL_RESPONSE_RE.fullmatch(str(text or ""))
    if (
        not match
        or not match.group("think").strip()
        or not match.group("proof").strip()
    ):
        return None
    return (
        match.group("think").strip(),
        match.group("proof").strip(),
        match.group("answer"),
    )

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent


ANSWER_RE = re.compile(r"ANSWER:\s*(True|False)\b", re.IGNORECASE)


SPACE_RE = re.compile(r"\s+")


def norm_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "").strip().lower().rstrip("."))


def extract_json_array(text: str) -> Optional[list]:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except Exception:
                    return None
                if isinstance(obj, list):
                    return obj
                return None
    return None


def parse_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        return normalize_boolean(m.group(1))
    except Exception:
        return None


def final_node_value(graph: list[dict[str, Any]] | None) -> Optional[str]:
    if not graph:
        return None
    for node in reversed(graph):
        if not isinstance(node, dict):
            continue
        if str(node.get("id", "")).lower() == "conclusion":
            val = str(node.get("value") or "").strip()
            return val or None
    for node in reversed(graph):
        if isinstance(node, dict) and str(node.get("type")) == "op":
            val = str(node.get("value") or "").strip()
            if val:
                return val
    for node in reversed(graph):
        if isinstance(node, dict) and str(node.get("type")) == "const":
            val = str(node.get("value") or "").strip()
            if val:
                return val
    return None


def graph_implied_answer(graph: list | None, question: str) -> Optional[str]:
    """Infer True/False from whether the final proved literal matches the question."""
    value = final_node_value(graph if isinstance(graph, list) else None)
    if value is None:
        return None
    q = norm_text(question)
    v = norm_text(value)
    if not q or not v:
        return None
    if v == q:
        return "True"
    # common closed-world negation patterns
    if v == f"not {q}" or v == f"{q} is false":
        return "False"
    if q.startswith("not ") and v == q[4:]:
        return "False"
    # "X is not Y" vs question "X is Y"
    if " not " in v and v.replace(" not ", " ") == q:
        return "False"
    if " not " in q and q.replace(" not ", " ") == v:
        return "False"
    # otherwise treat mismatch as False proof of complement / different literal
    return "False"


def gold_answer(item: dict) -> str:
    return normalize_boolean(item.get("answer"))


def gold_graph_lines(item: dict) -> list[dict[str, Any]]:
    try:
        return to_entailment_graph(item.get("canonical_graph"))
    except Exception:
        return []


def build_chat_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{system_prompt}\n\n{user_text}\n"


def intervention_user(item: dict, graph: list[dict[str, Any]]) -> str:
    rendered = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{build_user_input(item)}\n\n"
        f"Given the following proof graph, output only:\n"
        f"{rendered}\n"
        f"ANSWER: True|False"
    )


def mutate_final_value(graph: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    cloned = json.loads(json.dumps(graph))
    if not cloned:
        return cloned
    # flip / corrupt the last op or conclusion value
    for node in reversed(cloned):
        if not isinstance(node, dict):
            continue
        if str(node.get("type")) == "op" or str(node.get("id")).lower() == "conclusion":
            val = str(node.get("value") or "unknown")
            node["value"] = f"NOT {val}" if not val.upper().startswith("NOT ") else val[4:].strip()
            return cloned
    if isinstance(cloned[-1], dict):
        cloned[-1]["value"] = f"mutated-{rng.randint(0, 10**6)}"
    return cloned


def aggregate(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    # F_strong is only defined where both F_ID and F_OOD are; re-average F_OOD
    # over that same subset so it is comparable to F_strong.
    aligned = [
        r["F_OOD"]
        for r in rows
        if r.get("F_strong") is not None and r.get("F_OOD") is not None
    ]
    n_fid = sum(r.get("F_ID") is not None for r in rows)
    n_food = sum(r.get("F_OOD") is not None for r in rows)
    return {
        "n_samples": n,
        "n_f_id_valid": n_fid,
        "n_f_ood_valid": n_food,
        "n_f_strong_valid": sum(r.get("F_strong") is not None for r in rows),
        "F_ID": mean("F_ID"),
        "F_OOD": mean("F_OOD"),
        "F_OOD_aligned": (
            sum(1 for v in aligned if v) / len(aligned) if aligned else 0.0
        ),
        "F_strong": mean("F_strong"),
        "answer_accuracy": mean("answer_correct"),
        "answer_parse_rate": mean("answer_parsed"),
        "graph_parse_rate": mean("graph_parsed"),
        "graph_correct_rate": mean("graph_correct"),
        "n_COR": sum(r.get("intervention_type") == "COR" for r in rows),
        "n_CNF": sum(r.get("intervention_type") == "CNF" for r in rows),
    }


def online(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    system_prompt = get_system_prompt("graph_sft")  # match graph_sft / graph_grpo training
    rng = random.Random(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # {"train": [...], "test": [...]} bundle as shipped in datasets/
        if args.split not in data:
            raise KeyError(
                f"Split '{args.split}' not in {args.test_file}; available: {sorted(data)}"
            )
        data = data[args.split]
    if args.num_samples and args.num_samples > 0:
        idxs = list(range(len(data)))
        rng.shuffle(idxs)
        data = [data[i] for i in idxs[: args.num_samples]]

    use_lora = bool(getattr(args, "checkpoint_path", "") or "")
    tokenizer_path = args.checkpoint_path if use_lora else args.model_base_path
    if use_lora and not (Path(args.checkpoint_path) / "tokenizer_config.json").exists():
        tokenizer_path = args.model_base_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    maybe_set_gemma_chat_template(tokenizer, args.model_base_path)
    if use_lora:
        maybe_set_gemma_chat_template(tokenizer, args.checkpoint_path)

    llm_kwargs = dict(
        model=args.model_base_path,
        tokenizer=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        enable_lora=use_lora,
        max_lora_rank=args.max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    lora_request = (
        LoRARequest("proofwriter", 1, args.checkpoint_path) if use_lora else None
    )
    gen_kwargs = {"use_tqdm": False}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    full_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed
    )
    ans_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.answer_max_new_tokens, seed=args.seed
    )

    prompts = []
    meta = []
    for item in data:
        user = build_user_input(item)
        prompts.append(build_chat_prompt(tokenizer, system_prompt, user))
        meta.append(item)

    outputs = []
    bs = max(1, args.batch_size)
    for start in range(0, len(prompts), bs):
        chunk = prompts[start : start + bs]
        outputs.extend(llm.generate(chunk, full_sampling, **gen_kwargs))

    per_sample = []
    intervention_prompts = []
    intervention_meta = []

    for item, out in zip(meta, outputs):
        response = out.outputs[0].text
        pred = parse_answer(response)
        graph = extract_json_array(response)
        graph_ok = isinstance(graph, list) and bool(graph)
        g_ans = graph_implied_answer(graph, str(item.get("question", ""))) if graph_ok else None
        gold = gold_answer(item)
        fid = (pred == g_ans) if (pred is not None and g_ans is not None) else None
        # graph "correct" ≈ proves the gold boolean via implied answer
        graph_correct = (g_ans == gold) if g_ans is not None else False

        if graph_correct:
            itype = "CNF"
            mut = mutate_final_value(graph or [], rng)
            int_user = intervention_user(item, mut)
            expected_exec = graph_implied_answer(mut, str(item.get("question", "")))
        else:
            itype = "COR"
            ggraph = gold_graph_lines(item)
            int_user = intervention_user(item, ggraph) if ggraph else None
            expected_exec = gold if ggraph else None

        row = {
            "id": item.get("id"),
            "question": item.get("question"),
            "gold_answer": gold,
            "pred_answer": pred,
            "graph_answer": g_ans,
            "graph_parsed": graph_ok,
            "answer_parsed": pred is not None,
            "answer_correct": (pred == gold) if pred is not None else False,
            "graph_correct": graph_correct,
            "F_ID": fid,
            "F_OOD": None,
            "F_strong": None,
            "intervention_type": itype if int_user else None,
            "response": response,
        }
        per_sample.append(row)
        if int_user is not None and expected_exec is not None:
            intervention_prompts.append(
                build_chat_prompt(tokenizer, system_prompt, int_user)
            )
            intervention_meta.append((len(per_sample) - 1, expected_exec))

    if intervention_prompts:
        int_outputs = []
        for start in range(0, len(intervention_prompts), bs):
            chunk = intervention_prompts[start : start + bs]
            int_outputs.extend(llm.generate(chunk, ans_sampling, **gen_kwargs))
        for (idx, expected), out in zip(intervention_meta, int_outputs):
            pred_int = parse_answer(out.outputs[0].text)
            food = (pred_int == expected) if pred_int is not None else None
            per_sample[idx]["F_OOD"] = food
            fid = per_sample[idx]["F_ID"]
            per_sample[idx]["F_strong"] = (
                bool(fid) and bool(food) if (fid is not None and food is not None) else None
            )

    metrics = aggregate(per_sample)
    metrics.update(
        {
            "mode": "online_lora_graph" if use_lora else "online_icl_graph",
            "dataset_name": "proofwriter",
            "model_base_path": args.model_base_path,
            "checkpoint_path": args.checkpoint_path or "",
            "test_file": args.test_file,
            "num_samples_requested": args.num_samples,
            "n_evaluated": len(per_sample),
            "ood_protocol": "COR_if_incorrect_else_CNF",
        }
    )
    return {"metrics": metrics, "results": per_sample}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-base-path", required=True)
    p.add_argument(
        "--checkpoint-path",
        default="",
        help="LoRA adapter dir; empty = base-model ICL",
    )
    p.add_argument("--test-file", required=True)
    p.add_argument(
        "--split",
        default="test",
        help='Split to read when --test-file is a {"train": ..., "test": ...} bundle',
    )
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--output-stem", default="proofwriter_icl_graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--answer-max-new-tokens", type=int, default=128)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = online(args)
    out = out_dir / f"{args.output_stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
