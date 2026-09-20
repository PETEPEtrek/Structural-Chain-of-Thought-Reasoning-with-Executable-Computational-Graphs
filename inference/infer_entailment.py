#!/usr/bin/env python3
"""
vLLM inference for EntailmentBank graph generation with repository-style evaluation.

Generates a proof graph, converts it to EntailmentBank proof format,
and computes per-sample + aggregated metrics:
- leaves: F1 / AllCorrect
- steps: F1 / AllCorrect
- intermediates: BLEURT_F1 / BLEURT_AllCorrect
- overall: AllCorrect

Output JSON includes:
- summary (run config, token stats, aggregated metrics)
- results (per-example records and metrics)
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
ENTAILMENT_BANK_DIR = REPO_ROOT / "entailment_bank"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if ENTAILMENT_BANK_DIR.exists() and str(ENTAILMENT_BANK_DIR) not in sys.path:
    # Needed because entailment_bank/utils/eval_utils.py uses `from utils...` imports.
    sys.path.insert(0, str(ENTAILMENT_BANK_DIR))

try:
    from entailment_graph_prompts import get_system_prompt_entailment_graph
except ImportError:
    # Fallback so this script can run standalone from a shared bundle.
    _SYSTEM_PROMPT_ENTAILMENT_GRAPH = """You are an expert at building entailment graphs.

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
"""

    _SYSTEM_PROMPT_TASK2_ADDENDUM = """
Task-2 note (distractors):
- Context includes many irrelevant sentences.
- Use only the minimum relevant sent* facts needed for the proof.
- Do not include distractor-only derivations.
"""

    _SYSTEM_PROMPT_TASK3_ADDENDUM = """
Task-3 note (retrieval setting):
- Context is a retrieved subset from a larger corpus and may be noisy/incomplete.
- Build the best valid proof using only provided sent* facts.
- Do not invent missing facts outside the context.
"""

    def get_system_prompt_entailment_graph(task_id: int) -> str:
        if task_id == 2:
            return _SYSTEM_PROMPT_ENTAILMENT_GRAPH + "\n" + _SYSTEM_PROMPT_TASK2_ADDENDUM
        if task_id == 3:
            return _SYSTEM_PROMPT_ENTAILMENT_GRAPH + "\n" + _SYSTEM_PROMPT_TASK3_ADDENDUM
        return _SYSTEM_PROMPT_ENTAILMENT_GRAPH

LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*=\s*(?:entail_and\((.*?)\)|(.*?))\s*(?::: ?(.*))?\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--task-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--test-file", required=True, help="Gold slot file: jsonl or json.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results_entailment_vllm")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--filter-valid", action="store_true")
    parser.add_argument(
        "--strict-hypothesis-input",
        action="store_true",
        help=(
            "In generated graph, force 'hypothesis' to be terminal only. "
            "Any usage of hypothesis on RHS inputs is treated as invalid and removed."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--bleurt-checkpoint",
        default=None,
        help=(
            "Path to BLEURT checkpoint (bleurt-large-512 recommended) for "
            "repo-consistent intermediates/overall metrics."
        ),
    )
    return parser.parse_args()


def to_float_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_sanitize(value.item())
        except Exception:
            return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def load_gold_records(path: Path) -> List[Dict[str, Any]]:
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
    raise ValueError(f"Unsupported gold format in {path}. Expected jsonl or list json.")


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


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def parse_json_graph_nodes_from_response(text: str) -> List[Dict[str, Any]]:
    t = _strip_fences(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("computational_graph"), list):
            obj = obj.get("computational_graph")
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
    except Exception:
        pass

    arr = extract_json_array_from_text(t)
    if isinstance(arr, list) and all(isinstance(x, dict) for x in arr):
        return arr
    return []


def _strip_invalid_hypothesis_inputs_json(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")) != "op":
            out.append(node)
            continue
        nid = str(node.get("id", "")).strip()
        refs = [str(x).strip() for x in (node.get("inputs") or []) if str(x).strip()]
        if nid.lower() != "hypothesis":
            refs = [r for r in refs if r.lower() != "hypothesis"]
        if not refs:
            continue
        new_node = dict(node)
        new_node["inputs"] = refs
        out.append(new_node)
    return out


def json_graph_nodes_to_proof(nodes: List[Dict[str, Any]], hypothesis_text: str) -> str:
    parsed_steps: List[Tuple[str, List[str], str]] = []
    has_hypothesis = False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")) != "op":
            continue
        lhs = str(node.get("id", "")).strip()
        refs = [str(x).strip() for x in (node.get("inputs") or []) if str(x).strip()]
        concl = str(node.get("value", "") or "").strip()
        if not lhs or not refs:
            continue
        if lhs.lower() == "hypothesis":
            lhs = "hypothesis"
            has_hypothesis = True
        parsed_steps.append((lhs, refs, concl))

    if not parsed_steps:
        return ""

    if not has_hypothesis:
        lhs, refs, concl = parsed_steps[-1]
        parsed_steps[-1] = ("hypothesis", refs, concl or hypothesis_text)

    steps = []
    for lhs, refs, conclusion in parsed_steps:
        lhs_str = " & ".join(refs)
        if lhs == "hypothesis":
            if not conclusion:
                conclusion = hypothesis_text
            steps.append(f"{lhs_str} -> hypothesis: {conclusion}")
        elif conclusion:
            steps.append(f"{lhs_str} -> {lhs}: {conclusion}")
        else:
            steps.append(f"{lhs_str} -> {lhs}")
    return "; ".join(steps)


def _split_refs(inside: str) -> List[str]:
    # Accept multiple legacy/new separators in model outputs.
    refs = [x.strip() for x in re.split(r"\s*(?:AND|&|,)\s*", inside) if x.strip()]
    fixed = []
    for ref in refs:
        # Some old checkpoints emit "sent2 sent5" instead of using a delimiter.
        if " " in ref:
            fixed.extend([x.strip() for x in ref.split() if x.strip()])
        else:
            fixed.append(ref)
    return fixed


def lines_to_proof(lines: List[str], hypothesis_text: str) -> str:
    parsed_steps: List[Tuple[str, List[str], str]] = []
    id_map: Dict[str, str] = {}
    int_counter = 1
    has_hypothesis = False

    for line in lines:
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        lhs_raw = m.group(1).strip()
        inside_old = (m.group(2) or "").strip()
        inside_new = (m.group(3) or "").strip()
        inside = inside_old if inside_old else inside_new
        conclusion = (m.group(4) or "").strip()
        refs = _split_refs(inside)
        if not refs:
            continue

        lhs_low = lhs_raw.lower()
        if lhs_low == "hypothesis":
            lhs_norm = "hypothesis"
            has_hypothesis = True
        elif lhs_low.startswith("int") and lhs_low[3:].isdigit():
            lhs_norm = lhs_low
        else:
            lhs_norm = f"int{int_counter}"
            int_counter += 1

        id_map[lhs_raw] = lhs_norm
        refs_norm = [id_map.get(r, r) for r in refs]
        parsed_steps.append((lhs_norm, refs_norm, conclusion))

    if not parsed_steps:
        return ""

    # If model never emitted an explicit hypothesis node, treat last step as final hypothesis.
    if not has_hypothesis:
        last_lhs, last_refs, last_concl = parsed_steps[-1]
        parsed_steps[-1] = ("hypothesis", last_refs, last_concl or hypothesis_text)

    steps = []
    for lhs, refs, conclusion in parsed_steps:
        lhs_str = " & ".join(refs)
        if lhs == "hypothesis":
            if not conclusion:
                conclusion = hypothesis_text
            steps.append(f"{lhs_str} -> hypothesis: {conclusion}")
        elif conclusion:
            steps.append(f"{lhs_str} -> {lhs}: {conclusion}")
        else:
            steps.append(f"{lhs_str} -> {lhs}")
    return "; ".join(steps)


def _strip_invalid_hypothesis_inputs(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        m = LINE_RE.match(line.strip())
        if not m:
            cleaned.append(line)
            continue
        lhs = m.group(1).strip().lower()
        inside_old = (m.group(2) or "").strip()
        inside_new = (m.group(3) or "").strip()
        inside = inside_old if inside_old else inside_new
        refs = _split_refs(inside)
        if lhs != "hypothesis":
            refs = [r for r in refs if r.lower() != "hypothesis"]
        if not refs:
            continue
        concl = (m.group(4) or "").strip()
        rhs = " & ".join(refs)
        cleaned.append(f"{m.group(1).strip()}={rhs} :: {concl}")
    return cleaned


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


def _f1_sets(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))


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


def collect_token_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "total": 0.0,
            "avg": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }
    n = len(values)
    total = float(sum(values))
    mean = total / n
    var = sum((x - mean) ** 2 for x in values) / n
    mn = float(min(values))
    mx = float(max(values))
    return {
        "total": total,
        "avg": mean,
        "std": float(var**0.5),
        "min": mn,
        "max": mx,
        "range": mx - mn,
    }


def main() -> None:
    args = parse_args()

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer, activations
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install: pip install vllm transformers peft tqdm"
        ) from e

    # Compatibility with remote code that expects this symbol.
    activations.PytorchGELUTanh = activations.GELUTanh

    use_fallback_eval = False
    try:
        # Preferred import path
        from entailment_bank.utils.eval_utils import score_prediction_whole_proof
    except ImportError:
        try:
            # Fallback for repo scripts that rely on `from utils...`
            from utils.eval_utils import score_prediction_whole_proof
        except ImportError:
            use_fallback_eval = True
            print(
                "[warn] entailment_bank evaluator not found; using built-in fallback metrics."
            )

    bleurt_scorer = None
    if args.bleurt_checkpoint:
        try:
            from bleurt import score as bleurt_score
        except ImportError as e:
            raise SystemExit(
                "BLEURT requested, but package is missing. Install bleurt."
            ) from e
        bleurt_scorer = bleurt_score.BleurtScorer(args.bleurt_checkpoint)

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    gold_path = Path(args.test_file)
    gold_records = load_gold_records(gold_path)
    if args.limit > 0:
        gold_records = gold_records[: args.limit]

    if args.filter_valid:
        gold_records = [r for r in gold_records if r.get("proof")]

    if not gold_records:
        raise SystemExit("No samples to process after filtering.")

    tokenizer_path = (
        args.checkpoint_path
        if args.checkpoint_path
        and (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_base_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    lora_request = None
    if args.checkpoint_path:
        lora_request = LoRARequest("adapter", 1, args.checkpoint_path)

    llm = LLM(
        model=args.model_base_path,
        tokenizer=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        enable_lora=bool(args.checkpoint_path),
        max_lora_rank=args.max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_tag = Path(args.model_base_path).name.lower().replace("-", "_").replace(".", "_")
    ckpt_tag = (
        Path(args.checkpoint_path).name.lower().replace("-", "_").replace(".", "_")
        if args.checkpoint_path
        else "base_model"
    )
    dataset_tag = gold_path.stem.lower().replace("-", "_").replace(".", "_")
    out_name = (
        f"inference_vllm_entailment_graph_{model_tag}_{ckpt_tag}_"
        f"{dataset_tag}_{args.split}_seed{args.seed}.json"
    )
    if args.results_suffix:
        out_name = out_name[:-5] + args.results_suffix + ".json"
    out_path = results_dir / out_name

    metric_keys = [
        "proof-leaves",
        "proof-steps",
        "proof-intermediates",
        "proof-overall",
    ]
    metric_sums = {
        "leaves_f1": 0.0,
        "leaves_acc": 0.0,
        "steps_f1": 0.0,
        "steps_acc": 0.0,
        "interm_f1": 0.0,
        "interm_acc": 0.0,
        "overall_acc": 0.0,
    }

    prompt_tokens_values: List[int] = []
    completion_tokens_values: List[int] = []
    total_tokens_values: List[int] = []

    results: List[Dict[str, Any]] = []
    total = 0

    proof_eval_name = (
        "entail_whole_proof_align_eval_onlyIR"
        if args.task_id == 3
        else "entail_whole_proof_align_eval"
    )

    for i, rec in enumerate(tqdm(gold_records, desc="vLLM inference")):
        prompt = build_prompt(tokenizer, args.task_id, rec)
        out = llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )[0]
        completion = out.outputs[0]
        response_text = completion.text

        prompt_tokens = len(out.prompt_token_ids or [])
        completion_tokens = len(completion.token_ids or [])
        total_tokens = prompt_tokens + completion_tokens

        prompt_tokens_values.append(prompt_tokens)
        completion_tokens_values.append(completion_tokens)
        total_tokens_values.append(total_tokens)

        model_graph = parse_json_graph_nodes_from_response(response_text)
        if args.strict_hypothesis_input:
            model_graph = _strip_invalid_hypothesis_inputs_json(model_graph)
        model_proof = json_graph_nodes_to_proof(
            model_graph, hypothesis_text=rec.get("hypothesis", "")
        )

        pred_for_eval = {
            "id": rec.get("id", str(i)),
            "prediction": model_proof,
            "slots": {
                "proof": model_proof,
                "hypothesis": rec.get("hypothesis", ""),
            },
        }

        if use_fallback_eval:
            metrics = _fallback_score_prediction_whole_proof(
                prediction=pred_for_eval,
                gold=rec,
            )
        else:
            metrics = score_prediction_whole_proof(
                prediction=pred_for_eval,
                gold=rec,
                prediction_json=rec,
                dataset="entailment_bank",
                scoring_spec={
                    "hypothesis_eval": "nlg",
                    "proof_eval": proof_eval_name,
                },
                bleurt_scorer=bleurt_scorer,
            )

        for mk in metric_keys:
            if mk not in metrics:
                continue
        leaves_f1 = to_float_or_none((metrics.get("proof-leaves") or {}).get("F1")) or 0.0
        leaves_acc = to_float_or_none((metrics.get("proof-leaves") or {}).get("acc")) or 0.0
        steps_f1 = to_float_or_none((metrics.get("proof-steps") or {}).get("F1")) or 0.0
        steps_acc = to_float_or_none((metrics.get("proof-steps") or {}).get("acc")) or 0.0
        interm_f1 = (
            to_float_or_none((metrics.get("proof-intermediates") or {}).get("BLEURT_F1"))
            or 0.0
        )
        interm_acc = (
            to_float_or_none((metrics.get("proof-intermediates") or {}).get("BLEURT_acc"))
            or 0.0
        )
        overall_acc = to_float_or_none((metrics.get("proof-overall") or {}).get("acc")) or 0.0

        metric_sums["leaves_f1"] += leaves_f1
        metric_sums["leaves_acc"] += leaves_acc
        metric_sums["steps_f1"] += steps_f1
        metric_sums["steps_acc"] += steps_acc
        metric_sums["interm_f1"] += interm_f1
        metric_sums["interm_acc"] += interm_acc
        metric_sums["overall_acc"] += overall_acc

        total += 1

        results.append(
            {
                "index": float(i),
                "split": args.split,
                "dataset_index": None,
                "id": rec.get("id"),
                "question": rec.get("question", ""),
                "answer": rec.get("answer", ""),
                "hypothesis": rec.get("hypothesis", ""),
                "context": rec.get("context", "") or context_from_triples(rec),
                "ground_truth_proof": rec.get("proof", ""),
                "model_response": response_text,
                "model_graph": model_graph,
                "model_proof": model_proof,
                "metrics": {
                    "proof-leaves": {"F1": leaves_f1, "AllCorrect": leaves_acc},
                    "proof-steps": {"F1": steps_f1, "AllCorrect": steps_acc},
                    "proof-intermediates": {
                        "F1": interm_f1,
                        "AllCorrect": interm_acc,
                    },
                    "proof-overall": {"AllCorrect": overall_acc},
                },
                "is_correct": bool(overall_acc == 1.0),
                "token_usage": {
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "total_tokens": float(total_tokens),
                },
            }
        )

    prompt_stats = collect_token_stats(prompt_tokens_values)
    completion_stats = collect_token_stats(completion_tokens_values)
    total_stats = collect_token_stats(total_tokens_values)

    denom = float(total) if total > 0 else 1.0
    summary = {
        "mode": "graph_entailment_vllm",
        "model_base_path": args.model_base_path,
        "checkpoint_path": args.checkpoint_path,
        "test_file": args.test_file,
        "split": args.split,
        "task_id": float(args.task_id),
        "filter_valid": bool(args.filter_valid),
        "total": float(total),
        "overall_correct": float(metric_sums["overall_acc"]),
        "overall_accuracy": float(metric_sums["overall_acc"] / denom),
        "leaves_f1": float(metric_sums["leaves_f1"] / denom),
        "leaves_allcorrect": float(metric_sums["leaves_acc"] / denom),
        "steps_f1": float(metric_sums["steps_f1"] / denom),
        "steps_allcorrect": float(metric_sums["steps_acc"] / denom),
        "intermediates_f1": float(metric_sums["interm_f1"] / denom),
        "intermediates_allcorrect": float(metric_sums["interm_acc"] / denom),
        "results_folder": out_name,
        "seed": float(args.seed),
        "proof_eval": proof_eval_name,
        "eval_backend": "fallback_internal" if use_fallback_eval else "entailment_bank_repo",
        "bleurt_checkpoint": args.bleurt_checkpoint,
        "token_usage": {
            "prompt_tokens_total": prompt_stats["total"],
            "completion_tokens_total": completion_stats["total"],
            "total_tokens_total": total_stats["total"],
            "prompt_tokens_avg": prompt_stats["avg"],
            "completion_tokens_avg": completion_stats["avg"],
            "total_tokens_avg": total_stats["avg"],
            "prompt_tokens_std": prompt_stats["std"],
            "completion_tokens_std": completion_stats["std"],
            "total_tokens_std": total_stats["std"],
            "prompt_tokens_min": prompt_stats["min"],
            "completion_tokens_min": completion_stats["min"],
            "total_tokens_min": total_stats["min"],
            "prompt_tokens_max": prompt_stats["max"],
            "completion_tokens_max": completion_stats["max"],
            "total_tokens_max": total_stats["max"],
            "prompt_tokens_range": prompt_stats["range"],
            "completion_tokens_range": completion_stats["range"],
            "total_tokens_range": total_stats["range"],
        },
    }

    payload = {"summary": summary, "results": results}
    payload = json_sanitize(payload)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print("Aggregated metrics:")
    print(
        "  Leaves F1={:.2f} AllCorrect={:.2f}".format(
            100 * float(summary["leaves_f1"]),
            100 * float(summary["leaves_allcorrect"]),
        )
    )
    print(
        "  Steps  F1={:.2f} AllCorrect={:.2f}".format(
            100 * float(summary["steps_f1"]),
            100 * float(summary["steps_allcorrect"]),
        )
    )
    print(
        "  Interm F1={:.2f} AllCorrect={:.2f}".format(
            100 * float(summary["intermediates_f1"]),
            100 * float(summary["intermediates_allcorrect"]),
        )
    )
    print("  Overall AllCorrect={:.2f}".format(100 * float(summary["overall_accuracy"])))


if __name__ == "__main__":
    main()
