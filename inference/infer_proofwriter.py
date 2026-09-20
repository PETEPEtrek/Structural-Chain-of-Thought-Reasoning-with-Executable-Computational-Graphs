#!/usr/bin/env python3
"""
ProofWriter inference with vLLM: answer accuracy plus EntailmentBank-style
leaves/steps proof metrics for the graph methods.

Standalone: imports only pip packages, never another file in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Inlined from the original proofwriter_prompts module so this file stands alone.
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
# Inference and scoring
# --------------------------------------------------------------------------- #

ANSWER_LINE_RE = re.compile(r"(?im)^\s*ANSWER:\s*(True|False)\s*$")
ANSWER_ANY_RE = re.compile(r"(?i)\bANSWER:\s*(True|False)\b")
PROOF_LINE_RE = re.compile(r"(?im)^\s*PROOF:\s*(?P<proof>.+?)\s*$")
SOLUTION_RE = re.compile(r"(?is)<solution>\s*(True|False)\s*</solution>")
GRAPH_ANSWER_RE = re.compile(
    r"\A\s*(?P<graph>\[.*\])\s*\nANSWER: (?P<answer>True|False)\s*\Z", flags=re.DOTALL
)

GRAPH_METHODS = frozenset({"graph_sft", "graph_grpo"})
SPACE_RE = re.compile(r"\s+")


def norm_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip()).casefold()


def load_rows(path: str, split: str, num_samples: int) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data[split] if isinstance(data, dict) else data
    return rows[:num_samples] if num_samples > 0 else rows


def resolve_dtype(model_path: str, dtype: str) -> str:
    requested = (dtype or "bfloat16").lower()
    if requested in ("auto", "default"):
        requested = "bfloat16"
    name = str(model_path).lower()
    if ("gemma-2" in name or "gemma2" in name) and requested in ("float16", "fp16", "half"):
        print(f"Note: {model_path} -> bfloat16 (gemma2 rejects float16)", flush=True)
        return "bfloat16"
    return requested


def parse_graph_response(text: Any) -> Optional[list]:
    """Graph methods must emit exactly one JSON array followed by an ANSWER line."""
    match = GRAPH_ANSWER_RE.fullmatch(str(text or ""))
    if not match:
        return None
    try:
        graph = json.loads(match.group("graph"))
    except json.JSONDecodeError:
        return None
    if not isinstance(graph, list) or not graph:
        return None
    for node in graph:
        if not isinstance(node, dict) or "id" not in node or "type" not in node:
            return None
    return graph


def parse_predicted_answer(text: Any, method: str) -> Optional[str]:
    response = str(text or "")
    if method == "default_grpo":
        parsed = parse_tagged_response_with_proof(response)
        if parsed is not None:
            return normalize_boolean(parsed[2])
        match = SOLUTION_RE.search(response)
        if match:
            return normalize_boolean(match.group(1))
    if method in GRAPH_METHODS and parse_graph_response(response) is None:
        return None
    match = ANSWER_LINE_RE.search(response) or ANSWER_ANY_RE.search(response)
    return normalize_boolean(match.group(1)) if match else None


def extract_predicted_proof(text: Any, method: str) -> Optional[str]:
    response = str(text or "")
    if method == "default_grpo":
        parsed = parse_tagged_response_with_proof(response)
        if parsed is not None:
            return parsed[1]
    match = PROOF_LINE_RE.search(response)
    return match.group("proof").strip() if match else None


def gold_graphs(row: dict) -> list[list[dict]]:
    """Every gold proof graph for the row, de-duplicated by canonical signature."""
    candidates = [row.get("canonical_graph")]
    candidates.extend(
        proof.get("graph")
        for proof in (row.get("gold_proofs") or [])
        if isinstance(proof, dict)
    )
    result: list[list[dict]] = []
    seen: set[str] = set()
    for graph in candidates:
        if not isinstance(graph, (dict, list)):
            continue
        try:
            compact = to_entailment_graph(graph)
        except (TypeError, ValueError):
            continue
        signature = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            result.append(compact)
    return result


def eb_leaves_steps_from_graph(graph: list[dict]) -> tuple[set, set]:
    """EntailmentBank-style decomposition: leaf fact ids and (rule, inputs) steps."""
    leaves = {str(node["id"]) for node in graph if node.get("type") == "const"}
    steps = set()
    for node in graph:
        if node.get("type") != "op":
            continue
        inputs = tuple(str(x) for x in (node.get("inputs") or []))
        steps.add((str(node.get("rule") or ""), inputs))
    return leaves, steps


def f1_sets(predicted: set, gold: set) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def score_eb_against_golds(
    predicted_graph: Optional[list[dict]], gold_alts: list[list[dict]]
) -> dict[str, Optional[float]]:
    """Best-matching gold proof wins, ranked by leaves F1 then steps F1."""
    keys = ("leaves_f1", "leaves_allcorrect", "steps_f1", "steps_allcorrect", "overall_allcorrect")
    if predicted_graph is None or not gold_alts:
        return dict.fromkeys(keys)

    pred_leaves, pred_steps = eb_leaves_steps_from_graph(predicted_graph)
    best = dict.fromkeys(keys, 0.0)
    for gold in gold_alts:
        gold_leaves, gold_steps = eb_leaves_steps_from_graph(gold)
        leaves_f1 = f1_sets(pred_leaves, gold_leaves)
        steps_f1 = f1_sets(pred_steps, gold_steps)
        leaves_ac = 1.0 if leaves_f1 == 1.0 else 0.0
        steps_ac = 1.0 if steps_f1 == 1.0 else 0.0
        overall = 1.0 if leaves_ac and steps_ac else 0.0
        better = (
            leaves_f1 > best["leaves_f1"]
            or (leaves_f1 == best["leaves_f1"] and steps_f1 > best["steps_f1"])
            or (
                leaves_f1 == best["leaves_f1"]
                and steps_f1 == best["steps_f1"]
                and overall > best["overall_allcorrect"]
            )
        )
        if better:
            best = {
                "leaves_f1": leaves_f1,
                "leaves_allcorrect": leaves_ac,
                "steps_f1": steps_f1,
                "steps_allcorrect": steps_ac,
                "overall_allcorrect": overall,
            }
    return best


def mean(values: list) -> Optional[float]:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def build_messages(row: dict, method: str, demos: list[dict]) -> list[dict]:
    """System prompt, optional few-shot demos, then the query."""
    messages = [{"role": "system", "content": get_system_prompt(method)}]
    for demo in demos:
        messages.append({"role": "user", "content": build_user_input(demo)})
        if method in GRAPH_METHODS:
            target = graph_target(demo)
        elif method == "cot_answer_sft":
            target = cot_answer_target(demo)
        elif method == "default_grpo":
            target = default_grpo_full_target(demo)
        else:
            target = answer_target(demo)
        messages.append({"role": "assistant", "content": target})
    messages.append({"role": "user", "content": build_user_input(row)})
    return messages


def render_prompt(tokenizer, row: dict, method: str, demos: list[dict]) -> str:
    messages = build_messages(row, method, demos)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Chat templates that reject the system role: fold it into the first user turn.
        system = messages[0]["content"]
        folded = []
        for i, message in enumerate(messages[1:]):
            content = message["content"]
            if i == 0:
                content = f"System instruction:\n{system}\n\n{content}"
            folded.append({"role": message["role"], "content": content})
        return tokenizer.apply_chat_template(
            folded, tokenize=False, add_generation_prompt=True
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProofWriter inference with vLLM (answer accuracy + EB proof metrics)."
    )
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="LoRA adapter path; leave empty for base-model / ICL runs",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["answer_sft", "cot_answer_sft", "default_grpo", "graph_sft", "graph_grpo"],
    )
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=0,
        help="Demos drawn from the train split; 0 = zero-shot (the SFT/GRPO setting)",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = full split")
    parser.add_argument("--results-dir", default="results_proofwriter")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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

    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        raise SystemExit("Install: pip install vllm transformers tqdm")

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    method = args.method
    rows = load_rows(args.test_file, args.split, args.num_samples)

    demos: list[dict] = []
    if args.few_shot_k > 0:
        train_rows = load_rows(args.test_file, args.train_split, 0)
        demos = random.Random(args.seed).sample(
            train_rows, min(args.few_shot_k, len(train_rows))
        )

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
    maybe_set_gemma_chat_template(tokenizer, args.model_base_path)

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

    generation_kwargs: dict[str, Any] = {"use_tqdm": False}
    if use_lora:
        generation_kwargs["lora_request"] = LoRARequest("proofwriter", 1, args.checkpoint_path)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    prompts = [render_prompt(tokenizer, row, method, demos) for row in rows]

    outputs = []
    size = max(1, args.batch_size)
    for start in tqdm(range(0, len(prompts), size), desc=f"ProofWriter {method}"):
        outputs.extend(
            llm.generate(prompts[start : start + size], sampling, **generation_kwargs)
        )

    results = []
    for row, output in zip(rows, outputs):
        response = output.outputs[0].text
        completion = output.outputs[0]

        try:
            gold = normalize_boolean(row.get("answer"))
        except ValueError:
            continue
        predicted = parse_predicted_answer(response, method)

        graph = parse_graph_response(response) if method in GRAPH_METHODS else None
        eb = score_eb_against_golds(graph, gold_graphs(row)) if method in GRAPH_METHODS else {}

        predicted_proof = extract_predicted_proof(response, method)
        try:
            gold_proof = proof_expression(row)
        except ValueError:
            gold_proof = None
        proof_exact = (
            norm_text(predicted_proof) == norm_text(gold_proof)
            if predicted_proof is not None and gold_proof is not None
            else None
        )

        results.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "depth": row.get("depth"),
                "gold_answer": gold,
                "model_response": response,
                "model_answer": predicted,
                "model_graph": graph,
                "model_proof": predicted_proof,
                "gold_proof": gold_proof,
                "answer_parsed": predicted is not None,
                "is_correct": predicted is not None and predicted == gold,
                "proof_exact_match": proof_exact,
                "eb_metrics": eb or None,
                "token_usage": {
                    "prompt_tokens": len(output.prompt_token_ids or []),
                    "completion_tokens": len(completion.token_ids or []),
                },
            }
        )

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    summary: dict[str, Any] = {
        "mode": "proofwriter",
        "method": method,
        "model_base_path": args.model_base_path,
        "checkpoint_path": args.checkpoint_path or None,
        "test_file": args.test_file,
        "split": args.split,
        "few_shot_k": args.few_shot_k,
        "seed": args.seed,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "answer_parse_rate": (
            sum(1 for r in results if r["answer_parsed"]) / total if total else 0.0
        ),
        "proof_exact_match": mean([r["proof_exact_match"] for r in results]),
        "completion_tokens_avg": mean(
            [r["token_usage"]["completion_tokens"] for r in results]
        ),
    }
    if method in GRAPH_METHODS:
        summary["graph_parse_rate"] = (
            sum(1 for r in results if r["model_graph"] is not None) / total if total else 0.0
        )
        for key in (
            "leaves_f1",
            "leaves_allcorrect",
            "steps_f1",
            "steps_allcorrect",
            "overall_allcorrect",
        ):
            summary[key] = mean(
                [(r["eb_metrics"] or {}).get(key) for r in results]
            )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    out_name = (
        f"inference_proofwriter_{method}_{model_tag}_{args.split}"
        f"_k{args.few_shot_k}_seed{args.seed}{args.results_suffix}.json"
    )
    out_path = results_dir / out_name
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
