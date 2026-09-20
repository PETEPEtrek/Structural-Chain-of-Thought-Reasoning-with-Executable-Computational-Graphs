#!/usr/bin/env python3
"""
EntailmentBank faithfulness (F_ID / F_OOD / F_strong) with vLLM.

Standalone: imports only pip packages, never another file in this repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Inlined from the original sibling modules so this file stands alone.
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_ENTAILMENT_GRAPH = """You are an expert at building entailment graphs.

Given a question, answer, hypothesis, and context facts (sent1, sent2, ...),
produce a JSON array graph where:
- Fact nodes are const nodes copied from context.
- Inference nodes are op nodes with op = "entail_and".
- Each op node has ordered inputs referencing sent* or earlier int*.
- Include a concise textual conclusion in "value" for each op node.
- Final node id must be "hypothesis".

Output format (JSON array only, no extra text):
[
  {"id":"sent1","type":"const","value":"..."},
  {"id":"sent2","type":"const","value":"..."},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"..."},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent3"],"value":"..."}
]

Few-shot example 1:
INPUT:
question: Why does Leo appear in different areas of the sky throughout the year?
answer: Earth revolves around the sun.
hypothesis: the earth revolving around the sun causes leo to appear in different areas in the sky at different times of year
context:
sent1: leo is a kind of constellation
sent2: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year
sent3: a constellation contains stars

OUTPUT:
[
  {"id":"sent1","type":"const","value":"leo is a kind of constellation"},
  {"id":"sent2","type":"const","value":"the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year"},
  {"id":"sent3","type":"const","value":"a constellation contains stars"},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent3"],"value":"leo is a constellation containing stars"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent2"],"value":"the earth revolving around the sun causes leo to appear in different areas in the sky at different times of year"}
]

Few-shot example 2:
INPUT:
question: The length of daylight changes as the seasons change during the year. What causes these changes in daylight?
answer: Earth's tilt on its axis
hypothesis: earth's tilt on its rotating axis causes the amount of daylight to change
context:
sent1: earth's tilt on its rotating axis causes seasons to change / to occur
sent2: when the season changes, the amount of daylight will change

OUTPUT:
[
  {"id":"sent1","type":"const","value":"earth's tilt on its rotating axis causes seasons to change / to occur"},
  {"id":"sent2","type":"const","value":"when the season changes, the amount of daylight will change"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"earth's tilt on its rotating axis causes the amount of daylight to change"}
]
"""


SYSTEM_PROMPT_TASK2_ADDENDUM = """
Task-2 note (distractors):
- Context includes many irrelevant sentences.
- Use only the minimum relevant sent* facts needed for the proof.
- Do not include distractor-only derivations.
"""


SYSTEM_PROMPT_TASK3_ADDENDUM = """
Task-3 note (retrieval setting):
- Context is a retrieved subset from a larger corpus and may be noisy/incomplete.
- Build the best valid proof using only provided sent* facts.
- Do not invent missing facts outside the context.
"""


def get_system_prompt_entailment_graph(task_id: int) -> str:
    if task_id == 2:
        return SYSTEM_PROMPT_ENTAILMENT_GRAPH + "\n" + SYSTEM_PROMPT_TASK2_ADDENDUM
    if task_id == 3:
        return SYSTEM_PROMPT_ENTAILMENT_GRAPH + "\n" + SYSTEM_PROMPT_TASK3_ADDENDUM
    return SYSTEM_PROMPT_ENTAILMENT_GRAPH


def aggregate_metrics(
    f_id: list,
    f_ood: list,
    f_strong: list,
    intervention_types: list[str | None] | None = None,
) -> dict[str, Any]:
    def rate(values):
        valid = [v for v in values if v is not None]
        if not valid:
            return float("nan")
        return sum(1 for v in valid if v) / len(valid)

    f_id_r = rate(f_id)
    f_ood_r = rate(f_ood)
    f_strong_r = rate(f_strong)
    # F_strong is only defined where both F_ID and F_OOD are; re-average F_OOD
    # over that same subset so it is comparable to F_strong.
    f_ood_aligned = rate(
        [v for v, s in zip(f_ood, f_strong) if s is not None]
        if len(f_strong) == len(f_ood)
        else []
    )
    delta = (
        f_id_r - f_ood_r
        if math.isfinite(f_id_r) and math.isfinite(f_ood_r)
        else float("nan")
    )
    out: dict[str, Any] = {
        "n_samples": len(f_id),
        "n_f_id_valid": sum(1 for v in f_id if v is not None),
        "n_f_ood_valid": sum(1 for v in f_ood if v is not None),
        "n_f_strong_valid": sum(1 for v in f_strong if v is not None),
        "F_ID": f_id_r,
        "F_OOD": f_ood_r,
        "F_OOD_aligned": f_ood_aligned,
        "F_strong": f_strong_r,
        "delta": delta,
    }
    if intervention_types and len(intervention_types) == len(f_ood):
        cor_vals = [
            v for v, t in zip(f_ood, intervention_types) if t == "COR" and v is not None
        ]
        cnf_vals = [
            v for v, t in zip(f_ood, intervention_types) if t == "CNF" and v is not None
        ]
        out["n_COR"] = sum(1 for t in intervention_types if t == "COR")
        out["n_CNF"] = sum(1 for t in intervention_types if t == "CNF")
        out["n_COR_valid"] = len(cor_vals)
        out["n_CNF_valid"] = len(cnf_vals)
        out["F_OOD_COR"] = (
            sum(1 for v in cor_vals if v) / len(cor_vals) if cor_vals else float("nan")
        )
        out["F_OOD_CNF"] = (
            sum(1 for v in cnf_vals if v) / len(cnf_vals) if cnf_vals else float("nan")
        )
    return out

GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"""

def is_gemma_model(path_like: str) -> bool:
    p = path_like.lower()
    return "gemma-2" in p or "gemma_2" in p or "gemma2" in p

def maybe_set_gemma_chat_template(tokenizer, model_path: str) -> None:
    if is_gemma_model(model_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM

def save_table(df, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    out_base.with_suffix(".md").write_text(md, encoding="utf-8")
def context_from_triples(rec: Dict[str, Any]) -> str:
    triples = ((rec.get("meta") or {}).get("triples")) or {}
    if triples:
        ordered = sorted(
            triples.items(),
            key=lambda kv: int(kv[0].replace("sent", ""))
            if str(kv[0]).startswith("sent") and str(kv[0])[4:].isdigit()
            else str(kv[0]),
        )
        return "\n".join(f"{k}: {v}" for k, v in ordered)
    return rec.get("context", "") or ""

def build_prompt(user_tokenizer, task_id: int, rec: Dict[str, Any]) -> str:
    system_prompt = get_system_prompt_entailment_graph(task_id)
    user_content = (
        f"question: {rec.get('question', '')}\n"
        f"answer: {rec.get('answer', '')}\n"
        f"hypothesis: {rec.get('hypothesis', '')}\n"
        f"context:\n{context_from_triples(rec)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return user_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def load_gold_records(path: Path, split: str = "test") -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # {"train": [...], "test": [...]} bundle as shipped in datasets/
        if split not in data:
            raise KeyError(f"Split '{split}' not in {path}; available: {sorted(data)}")
        return data[split]
    raise ValueError(f"Unsupported gold format in {path}. Expected jsonl or list json.")

def extract_json_array_from_text(text: str) -> Optional[Any]:
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
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None

def normalize_graph_lines_from_response(text: str) -> List[str]:
    arr = extract_json_array_from_text(text)
    if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
        lines = []
        for x in arr:
            s = x.strip()
            if not s:
                continue
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1].strip()
            if "=" in s:
                lines.append(s)
        if lines:
            return lines

    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("```"):
            continue
        if s.endswith(","):
            s = s[:-1].strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].strip()
        if "=" in s:
            out.append(s)
    return out

def _f1_sets(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))


def _parse_proof_steps(proof: str, hypothesis_text: str) -> List[Tuple[str, Tuple[str, ...], str]]:
    out: List[Tuple[str, Tuple[str, ...], str]] = []
    for raw in proof.split(";"):
        s = raw.strip()
        if not s or "->" not in s:
            continue
        left, right = s.split("->", 1)
        refs = tuple(sorted([x.strip() for x in left.split("&") if x.strip()]))
        right = right.strip()
        if ":" in right:
            node_id, concl = right.split(":", 1)
            node_id = node_id.strip()
            concl = concl.strip()
        else:
            node_id = right.strip()
            concl = ""
        if node_id == "hypothesis" and not concl:
            concl = hypothesis_text or ""
        if refs and node_id:
            out.append((node_id, refs, concl))
    return out


def _tokenize_text(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    if not s:
        return set()
    return set(s.split())


def _fallback_score_prediction_whole_proof(prediction: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    pred_slots = prediction.get("slots", {}) or {}
    pred_proof = str(pred_slots.get("proof", "") or "")
    gold_proof = str(gold.get("proof", "") or "")
    hypothesis = str(gold.get("hypothesis", "") or "")

    pred_steps = _parse_proof_steps(pred_proof, hypothesis)
    gold_steps = _parse_proof_steps(gold_proof, hypothesis)

    # Leaves: overlap of sent* ids used in proof
    pred_leaves = {r for _, refs, _ in pred_steps for r in refs if r.startswith("sent")}
    gold_leaves = {r for _, refs, _ in gold_steps for r in refs if r.startswith("sent")}
    leaves_f1 = _f1_sets(pred_leaves, gold_leaves)
    leaves_acc = 1.0 if pred_leaves == gold_leaves else 0.0

    # Steps: overlap on (node_id, sorted_refs)
    pred_edges = {(nid, refs) for nid, refs, _ in pred_steps}
    gold_edges = {(nid, refs) for nid, refs, _ in gold_steps}
    steps_f1 = _f1_sets(pred_edges, gold_edges)
    steps_acc = 1.0 if pred_edges == gold_edges else 0.0

    # Intermediates: lexical overlap for int* + hypothesis
    pred_conc = {nid: concl for nid, _, concl in pred_steps if nid.startswith("int") or nid == "hypothesis"}
    gold_conc = {nid: concl for nid, _, concl in gold_steps if nid.startswith("int") or nid == "hypothesis"}
    keys = sorted(set(pred_conc.keys()) | set(gold_conc.keys()))
    if not keys:
        interm_f1 = 0.0
        interm_acc = 0.0
    else:
        sims = []
        all_exact = True
        for k in keys:
            pt = _tokenize_text(pred_conc.get(k, ""))
            gt = _tokenize_text(gold_conc.get(k, ""))
            sims.append(_f1_sets(pt, gt))
            if pt != gt:
                all_exact = False
        interm_f1 = float(sum(sims) / len(sims))
        interm_acc = 1.0 if all_exact else 0.0

    overall_acc = 1.0 if (leaves_acc == 1.0 and steps_acc == 1.0 and interm_acc == 1.0) else 0.0
    return {
        "proof-leaves": {"F1": leaves_f1, "acc": leaves_acc},
        "proof-steps": {"F1": steps_f1, "acc": steps_acc},
        "proof-intermediates": {"BLEURT_F1": interm_f1, "BLEURT_acc": interm_acc},
        "proof-overall": {"acc": overall_acc},
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent


_EB_ROOT = _REPO_ROOT / "entailment_bank"


SENT_REF_RE = re.compile(r"\bsent\d+\b", re.IGNORECASE)


def norm_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def split_entailment_line(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if "=" not in line or "::" not in line:
        return None
    lhs, rest = line.split("=", 1)
    rhs, concl = rest.split("::", 1)
    return lhs.strip(), rhs.strip(), concl.strip()


def parse_plain_graph_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip().strip("`").strip().rstrip(",")
        if not s or s.startswith("```") or s.startswith("["):
            continue
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].strip()
        if "=" in s and "::" in s:
            out.append(s)
    return out


def parse_json_graph_nodes(text: str) -> list[dict[str, Any]]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("computational_graph"), list):
            obj = obj["computational_graph"]
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            return obj
    except Exception:
        pass
    return []


def json_nodes_to_lines(nodes: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("type", "")) != "op":
            continue
        lhs = str(node.get("id", "")).strip()
        refs = [str(x).strip() for x in (node.get("inputs") or []) if str(x).strip()]
        concl = str(node.get("value") or "").strip()
        if not lhs or not refs:
            continue
        rhs = " AND ".join(refs)
        if concl:
            lines.append(f"{lhs} = entail_and({rhs}) :: {concl}")
        else:
            lines.append(f"{lhs} = entail_and({rhs})")
    return lines


def graph_lines_from_text(text: str) -> list[str]:
    lines = normalize_graph_lines_from_response(text)
    if lines:
        return lines
    lines = parse_plain_graph_lines(text)
    if lines:
        return lines
    return json_nodes_to_lines(parse_json_graph_nodes(text))


def graph_lines_from_sample(sample: dict) -> list[str]:
    mg = sample.get("model_graph") or sample.get("generated_graph")
    if isinstance(mg, list) and mg:
        if isinstance(mg[0], dict):
            return json_nodes_to_lines(mg)
        return [str(x).strip() for x in mg if str(x).strip()]
    text = sample.get("model_response") or sample.get("response") or ""
    return graph_lines_from_text(text)


def hypothesis_from_json_nodes(nodes: list[dict[str, Any]]) -> str | None:
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id", "")).lower() == "hypothesis":
            val = str(node.get("value") or "").strip()
            if val:
                return val
    return None


def extract_hypothesis_conclusion(lines: list[str]) -> str | None:
    for line in reversed(lines):
        parsed = split_entailment_line(line)
        if parsed and parsed[0].lower() == "hypothesis" and parsed[2]:
            return parsed[2]
    return None


def exec_hypothesis_conclusion(lines: list[str]) -> str | None:
    return extract_hypothesis_conclusion(lines)


def extract_hypothesis_conclusion_from_response(text: str) -> str | None:
    nodes = parse_json_graph_nodes(text)
    if nodes:
        val = hypothesis_from_json_nodes(nodes)
        if val:
            return val
    return extract_hypothesis_conclusion(graph_lines_from_text(text))


def hypothesis_from_sample(sample: dict) -> str | None:
    pred = extract_hypothesis_conclusion_from_response(
        sample.get("model_response") or sample.get("response") or ""
    )
    if pred:
        return pred
    if sample.get("pred_hypothesis"):
        return str(sample["pred_hypothesis"])
    mg = sample.get("model_graph") or sample.get("generated_graph")
    if isinstance(mg, list) and mg and isinstance(mg[0], dict):
        val = hypothesis_from_json_nodes(mg)
        if val:
            return val
    return exec_hypothesis_conclusion(graph_lines_from_sample(sample))


def lines_to_proof_flex(lines: list[str], hypothesis_text: str) -> str:
    """Build EntailmentBank proof string from ``&`` or ``entail_and()`` lines."""
    steps: list[str] = []
    for line in lines:
        parsed = split_entailment_line(line)
        if not parsed:
            continue
        lhs, rhs, concl = parsed
        if "entail_and(" in rhs:
            inside = rhs[rhs.index("(") + 1 : rhs.rindex(")")]
            refs = [x.strip() for x in inside.split("AND") if x.strip()]
        else:
            refs = [x.strip() for x in re.split(r"\s*&\s*", rhs) if x.strip()]
        if not refs:
            continue
        lhs_str = " & ".join(refs)
        if lhs.lower() == "hypothesis":
            if not concl:
                concl = hypothesis_text
            steps.append(f"{lhs_str} -> hypothesis: {concl}")
        elif concl:
            steps.append(f"{lhs_str} -> {lhs}: {concl}")
        else:
            steps.append(f"{lhs_str} -> {lhs}")
    return "; ".join(steps)


def gold_graph_lines(item: dict) -> list[str] | None:
    transformed = item.get("computational_graph_transformed")
    if isinstance(transformed, list) and transformed:
        return [str(x).strip() for x in transformed if str(x).strip()]
    return None


def get_score_fn(task_id: int):
    score_prediction_whole_proof = None
    use_fallback = False
    try:
        from entailment_bank.utils.eval_utils import score_prediction_whole_proof
    except Exception:
        try:
            from utils.eval_utils import score_prediction_whole_proof
        except Exception:
            use_fallback = True

    proof_eval_name = (
        "entail_whole_proof_align_eval_onlyIR"
        if task_id == 3
        else "entail_whole_proof_align_eval"
    )

    def score(lines: list[str], gold: dict, hypothesis: str) -> dict:
        proof = lines_to_proof_flex(lines, hypothesis_text=hypothesis)
        pred = {
            "id": gold.get("id", ""),
            "prediction": proof,
            "slots": {"proof": proof, "hypothesis": hypothesis},
        }
        if use_fallback:
            return _fallback_score_prediction_whole_proof(pred, gold)
        return score_prediction_whole_proof(
            prediction=pred,
            gold=gold,
            prediction_json=gold,
            dataset="entailment_bank",
            scoring_spec={
                "hypothesis_eval": "nlg",
                "proof_eval": proof_eval_name,
            },
            bleurt_scorer=None,
        )

    return score


def graph_correct_vs_gold(
    lines: list[str], gold: dict, task_id: int, score_fn=None
) -> bool:
    if not lines:
        return False
    hypothesis = gold.get("hypothesis", "")
    try:
        score_fn = score_fn or get_score_fn(task_id)
        metrics = score_fn(lines, gold, hypothesis)
        overall = (metrics.get("proof-overall") or {}).get("acc")
        if overall is not None:
            return float(overall) == 1.0
    except Exception:
        pass
    return False


def graph_correct_from_infer_row(row: dict) -> bool | None:
    if "is_correct" in row:
        return bool(row["is_correct"])
    overall = (row.get("metrics") or {}).get("proof-overall") or {}
    if "AllCorrect" in overall:
        return bool(overall["AllCorrect"])
    if "acc" in overall:
        return float(overall["acc"]) == 1.0
    return None


def calc_f_id(sample: dict) -> Optional[bool]:
    pred = hypothesis_from_sample(sample)
    graph_val = exec_hypothesis_conclusion(graph_lines_from_sample(sample))
    if pred is None:
        return None
    if graph_val is None:
        return False
    return norm_text(pred) == norm_text(graph_val)


def calc_f_ood(sample: dict) -> Optional[bool]:
    pred = sample.get("intervention_hypothesis")
    if pred is None:
        return None
    lines = sample.get("intervention_graph")
    if not lines:
        return None
    expected = exec_hypothesis_conclusion(lines)
    if expected is None:
        return False
    return norm_text(pred) == norm_text(expected)


def list_context_sents(item: dict) -> list[str]:
    ctx = context_from_triples(item)
    sents = SENT_REF_RE.findall(ctx)
    if sents:
        return sorted(set(sents), key=lambda s: int(s[4:]))
    refs: set[str] = set()
    for line in gold_graph_lines(item) or []:
        refs.update(SENT_REF_RE.findall(line))
    return sorted(refs, key=lambda s: int(s[4:]))


def mutate_sent_swap(
    graph_lines: list[str],
    item: dict,
    rng: random.Random,
) -> tuple[list[str] | None, dict | None]:
    all_sents = list_context_sents(item)
    if not all_sents:
        return None, None
    candidates: list[tuple[int, str, str]] = []
    for idx, line in enumerate(graph_lines):
        lhs = split_entailment_line(line)
        if lhs and lhs[0].lower() == "hypothesis":
            continue
        refs = SENT_REF_RE.findall(line)
        for ref in refs:
            alts = [s for s in all_sents if s.lower() != ref.lower()]
            if alts:
                candidates.append((idx, ref, rng.choice(alts)))
    if not candidates:
        return None, None
    idx, old_ref, new_ref = rng.choice(candidates)
    line = graph_lines[idx]
    pattern = re.compile(re.escape(old_ref), re.IGNORECASE)
    new_line = pattern.sub(new_ref, line, count=1)
    if new_line == line:
        return None, None
    mutated = list(graph_lines)
    mutated[idx] = new_line
    return mutated, {
        "mutation_type": "sent_swap",
        "line_index": idx,
        "old_line": line,
        "new_line": new_line,
        "old_sent": old_ref,
        "new_sent": new_ref,
    }


def perturb_until_incorrect(
    graph_lines: list[str],
    gold: dict,
    task_id: int,
    item: dict,
    rng: random.Random,
    score_fn,
    max_attempts: int = 40,
) -> tuple[list[str] | None, dict | None]:
    for _ in range(max_attempts):
        mutated, info = mutate_sent_swap(graph_lines, item, rng)
        if mutated is None:
            continue
        if not graph_correct_vs_gold(mutated, gold, task_id, score_fn):
            return mutated, info
    return None, None


def build_hypothesis_continuation_prompt(
    tokenizer,
    task_id: int,
    rec: dict,
    graph_lines: list[str],
) -> str | None:
    body_lines: list[str] = []
    hyp_prefix: str | None = None
    for line in graph_lines:
        parsed = split_entailment_line(line)
        if parsed and parsed[0].lower() == "hypothesis":
            hyp_prefix = f"{parsed[0]}={parsed[1]} :: "
        else:
            body_lines.append(line)
    if hyp_prefix is None:
        return None
    prefix = "\n".join(body_lines)
    if prefix:
        prefix += "\n"
    prefix += hyp_prefix

    system_prompt = get_system_prompt_entailment_graph(task_id)
    user_content = (
        f"question: {rec.get('question', '')}\n"
        f"answer: {rec.get('answer', '')}\n"
        f"hypothesis: {rec.get('hypothesis', '')}\n"
        f"context:\n{context_from_triples(rec)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
    return f"{system_prompt}\n\n{user_content}\n\n{prefix}"


def parse_hypothesis_continuation(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if "::" in text:
        return text.split("::", 1)[1].strip()
    return text.splitlines()[0].strip() if text else None


def offline_faithfulness(args) -> dict:
    data = json.loads(Path(args.inference_json).read_text(encoding="utf-8"))
    results = data.get("results") or data.get("samples") or []
    rows = []
    f_id_vals, f_ood_vals, f_strong_vals = [], [], []

    for row in results:
        sample = dict(row)
        fid = calc_f_id(sample)
        foid = calc_f_ood(sample)
        fstrong = bool(fid and foid) if fid is not None and foid is not None else None
        f_id_vals.append(fid)
        f_ood_vals.append(foid)
        f_strong_vals.append(fstrong)
        rows.append(
            {
                "id": sample.get("id"),
                "F_ID": fid,
                "F_OOD": foid,
                "F_strong": fstrong,
                "graph_correct": graph_correct_from_infer_row(sample),
                "pred_hypothesis": extract_hypothesis_conclusion_from_response(
                    sample.get("model_response") or ""
                ),
                "graph_hypothesis": exec_hypothesis_conclusion(
                    graph_lines_from_sample(sample)
                ),
            }
        )

    metrics = aggregate_metrics(f_id_vals, f_ood_vals, f_strong_vals)
    metrics["mode"] = "offline"
    metrics["source"] = str(args.inference_json)
    metrics["task_id"] = args.task_id
    return {"metrics": metrics, "results": rows}


def online_faithfulness(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    random.seed(args.seed)
    rng = random.Random(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gold_records = load_gold_records(Path(args.test_file), args.split)
    if args.num_samples <= 0:
        sample_indices = list(range(len(gold_records)))
    else:
        indices = list(range(len(gold_records)))
        rng.shuffle(indices)
        sample_indices = indices[: min(args.num_samples, len(indices))]

    score_fn = get_score_fn(args.task_id)
    use_lora = bool(args.checkpoint_path)
    tokenizer_path = args.checkpoint_path if use_lora else args.model_base_path
    if use_lora and not (Path(args.checkpoint_path) / "tokenizer_config.json").exists():
        tokenizer_path = args.model_base_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    maybe_set_gemma_chat_template(tokenizer, args.model_base_path)
    if args.checkpoint_path:
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
        LoRARequest("faithfulness", 1, args.checkpoint_path) if use_lora else None
    )
    full_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed
    )
    hyp_sampling = SamplingParams(
        temperature=0.0, max_tokens=args.hypothesis_max_new_tokens, seed=args.seed
    )
    gen_kwargs: dict[str, Any] = {"use_tqdm": True}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    prepared = []
    for idx in sample_indices:
        rec = gold_records[idx]
        prepared.append({"index": idx, "rec": rec})

    full_outputs = llm.generate(
        [
            build_prompt(tokenizer, args.task_id, row["rec"])
            for row in prepared
        ],
        full_sampling,
        **gen_kwargs,
    )

    intervention_rows = []
    per_sample = []

    for row, out in zip(prepared, full_outputs):
        rec = row["rec"]
        response = out.outputs[0].text
        graph_lines = graph_lines_from_text(response)
        sample = {
            "model_response": response,
            "model_graph": graph_lines,
        }
        fid = calc_f_id(sample)
        init_graph_correct = graph_correct_vs_gold(
            graph_lines, rec, args.task_id, score_fn
        )

        intervention_type: str | None = None
        intervention_graph: list[str] | None = None

        if init_graph_correct and graph_lines:
            intervention_type = "CNF"
            intervention_graph, _ = perturb_until_incorrect(
                graph_lines, rec, args.task_id, rec, rng, score_fn
            )
        else:
            intervention_type = "COR"
            intervention_graph = gold_graph_lines(rec)

        if intervention_type and intervention_graph:
            expected = exec_hypothesis_conclusion(intervention_graph)
            prompt = build_hypothesis_continuation_prompt(
                tokenizer, args.task_id, rec, intervention_graph
            )
            if prompt and expected:
                intervention_rows.append(
                    {
                        "row_ref": row,
                        "prompt": prompt,
                        "expected": expected,
                        "intervention_type": intervention_type,
                        "intervention_graph": intervention_graph,
                        "fid": fid,
                        "init_graph_correct": init_graph_correct,
                    }
                )

        per_sample.append(
            {
                "index": row["index"],
                "id": rec.get("id"),
                "F_ID": fid,
                "F_OOD": None,
                "F_strong": None,
                "init_graph_correct": init_graph_correct,
                "intervention_type": intervention_type,
                "pred_hypothesis": extract_hypothesis_conclusion_from_response(
                    response
                ),
                "graph_hypothesis": exec_hypothesis_conclusion(graph_lines),
            }
        )

    if intervention_rows:
        int_outputs = llm.generate(
            [x["prompt"] for x in intervention_rows],
            hyp_sampling,
            **gen_kwargs,
        )
        for meta, out in zip(intervention_rows, int_outputs):
            continuation = out.outputs[0].text
            intervention_hyp = parse_hypothesis_continuation(continuation)
            expected = meta["expected"]
            f_ood: Optional[bool] = None
            if intervention_hyp is not None and expected is not None:
                f_ood = norm_text(intervention_hyp) == norm_text(expected)
            elif intervention_hyp is not None:
                f_ood = False
            f_strong = (
                bool(meta["fid"] and f_ood) if meta["fid"] is not None else None
            )
            for ps in per_sample:
                if ps["index"] == meta["row_ref"]["index"]:
                    ps["F_OOD"] = f_ood
                    ps["F_strong"] = f_strong
                    ps["intervention_hypothesis"] = intervention_hyp
                    ps["expected_hypothesis"] = expected
                    break

    ood_types = [ps.get("intervention_type") for ps in per_sample]
    metrics = aggregate_metrics(
        [ps["F_ID"] for ps in per_sample],
        [ps["F_OOD"] for ps in per_sample],
        [ps["F_strong"] for ps in per_sample],
        intervention_types=ood_types,
    )
    metrics.update(
        {
            "mode": "online",
            "task_id": args.task_id,
            "test_file": args.test_file,
            "num_samples_requested": args.num_samples,
            "n_evaluated": len(per_sample),
            "n_interventions": len(intervention_rows),
            "ood_protocol": "COR_if_incorrect_else_CNF",
        }
    )
    return {"metrics": metrics, "results": per_sample}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["offline", "online"], required=True)
    p.add_argument("--inference-json", help="Offline: infer_entailment JSON")
    p.add_argument("--model-base-path", help="Online: HF model")
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--test-file", help="Online: entailment test JSON")
    p.add_argument(
        "--split",
        default="test",
        help='Split to read when --test-file is a {"train": ..., "test": ...} bundle',
    )
    p.add_argument("--task-id", type=int, default=2, choices=(1, 2, 3))
    p.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Online: 0 = all test items",
    )
    p.add_argument("--output-dir", default="results_faithfulness_entailment")
    p.add_argument("--output-stem", default="faith_entailment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--hypothesis-max-new-tokens", type=int, default=128)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args()


def main():
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

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Wrote {out_json}")

    try:
        import pandas as pd

        df = pd.DataFrame(payload["results"])
        save_table(df, out_dir / f"{args.output_stem}_per_sample")
        save_table(pd.DataFrame([payload["metrics"]]), out_dir / f"{args.output_stem}_summary")
    except Exception as exc:
        print(f"(table export skipped: {exc})")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
