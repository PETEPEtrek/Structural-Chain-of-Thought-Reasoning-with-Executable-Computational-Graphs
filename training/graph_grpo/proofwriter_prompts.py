"""Prompts and deterministic formatting for ProofWriter training."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

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

# Fixed demonstrations are deliberately shared by Graph SFT and Graph GRPO only.
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