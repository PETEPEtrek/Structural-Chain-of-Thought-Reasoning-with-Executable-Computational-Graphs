#!/usr/bin/env python3
"""
GRPO training for EntailmentBank entailment graphs.

Supports two target modes:
- graph_json: full JSON graph nodes (const/op with inputs/value)

Reward adaptation:
- proof-leaves: overlap on used sent* leaves
- proof-steps: overlap on step edges (lhs -> refs)
- proof-intermediates: text similarity of intermediate conclusions (Jaccard or BLEURT backend)
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, activations
from trl import GRPOConfig, GRPOTrainer

from entailment_graph_prompts import (
    get_system_prompt_entailment_graph,
)

# transformers>=4.52 removed PytorchGELUTanh symbol used by some remote model code.
activations.PytorchGELUTanh = activations.GELUTanh

LHS_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:entail_and\((.*?)\)|(.*?))\s*(?::: ?(.*))?$"
)

GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"""


def is_gemma_2_2b(path_like: str) -> bool:
    low = path_like.lower()
    return "gemma-2-2b" in low or "gemma_2_2b" in low


def maybe_set_gemma_chat_template(tokenizer, model_name: str) -> None:
    if is_gemma_2_2b(model_name):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--task-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=0,
        help="If >0, force gradient accumulation steps; if 0, auto-compute from num_generations.",
    )
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument(
        "--rewards",
        default="proof_leaves,proof_steps,proof_intermediates",
        help="Comma-separated from: proof_leaves,proof_steps,proof_intermediates",
    )
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    parser.add_argument("--debug-reward", action="store_true")
    parser.add_argument("--debug-reward-max-samples", type=int, default=1)
    parser.add_argument("--debug-reward-max-calls", type=int, default=200)
    parser.add_argument("--debug-reward-max-text", type=int, default=800)
    parser.add_argument(
        "--proof-intermediates-reward-backend",
        choices=("jaccard", "bleurt"),
        default="jaccard",
        help="Reward backend for proof_intermediates: jaccard (fast) or bleurt (closer to official eval).",
    )
    parser.add_argument(
        "--bleurt-checkpoint",
        default="",
        help="Local path to BLEURT checkpoint folder (required when backend=bleurt).",
    )
    parser.add_argument(
        "--bleurt-threshold",
        type=float,
        default=0.28,
        help="BLEURT threshold for node-level correctness (official eval default is ~0.28).",
    )
    parser.add_argument(
        "--bleurt-batch-size",
        type=int,
        default=1,
        help="Batch size for BLEURT scorer calls.",
    )
    parser.add_argument(
        "--debug-print-dataset-example",
        action="store_true",
        help="Print one full training example (prompt + gold graph fields) to logs.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Do not save adapter/tokenizer at the end (useful for smoke debugging).",
    )
    return parser.parse_args()


def parse_line(line: str):
    m = LHS_RE.match(line.strip())
    if not m:
        return None
    lhs = m.group(1).strip()
    inside_old = (m.group(2) or "").strip()
    inside_new = (m.group(3) or "").strip()
    inside = inside_old if inside_old else inside_new
    concl = (m.group(4) or "").strip()
    refs = [x.strip() for x in re.split(r"\s*(?:AND|&|,)\s*", inside) if x.strip()]
    fixed = []
    for ref in refs:
        if " " in ref:
            fixed.extend([x.strip() for x in ref.split() if x.strip()])
        else:
            fixed.append(ref)
    refs = fixed
    if not refs:
        return None
    return lhs, refs, concl


def graph_nodes_to_steps(graph_nodes) -> List[Tuple[str, List[str], str]]:
    steps = []
    if not isinstance(graph_nodes, list):
        return steps
    for node in graph_nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")) != "op":
            continue
        lhs = str(node.get("id", "")).strip()
        if not lhs:
            continue
        refs = [str(x).strip() for x in (node.get("inputs") or []) if str(x).strip()]
        if not refs:
            continue
        concl = str(node.get("value", "") or "")
        steps.append((lhs, refs, concl))
    return steps


def parse_json_graph_text(text: str):
    text = (text or "").strip()
    if not text:
        return []

    # Strip common markdown fences.
    text = text.replace("```json", "```").replace("```JSON", "```")
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    payload = None
    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except Exception:
                payload = None
    if isinstance(payload, dict) and isinstance(payload.get("computational_graph"), list):
        payload = payload.get("computational_graph")
    if not isinstance(payload, list):
        return []
    return payload


def extract_intermediate_texts_from_nodes(graph_nodes):
    out = {}
    if not isinstance(graph_nodes, list):
        return out
    for node in graph_nodes:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("id", ""))
        if not nid.startswith("int"):
            continue
        out[nid] = str(node.get("value", "") or "")
    return out


def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_sets(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))


def build_training_dataset(
    dataset_file: Path,
    task_id: int,
    allow_system_role: bool = True,
    debug_print_dataset_example: bool = False,
) -> Dataset:
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    system_prompt = get_system_prompt_entailment_graph(task_id)

    data = []
    for r in rows:
        q = r.get("question", "")
        a = r.get("answer", "")
        h = r.get("hypothesis", "")
        c = r.get("context", "")
        graph_lines = r.get("computational_graph_transformed", [])
        graph_nodes = r.get("computational_graph", [])
        intermediate_texts = extract_intermediate_texts_from_nodes(graph_nodes)
        if not q or not c:
            continue
        if not graph_nodes:
            continue

        user_content = f"question: {q}\nanswer: {a}\nhypothesis: {h}\ncontext: {c}"
        if allow_system_role:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            prompt = [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{user_content}",
                }
            ]
        data.append(
            {
                "prompt": prompt,
                "graph_lines": graph_lines,
                "graph_nodes": graph_nodes,
                "intermediate_texts": intermediate_texts,
                "id": r.get("id", ""),
            }
        )
    if not data:
        raise ValueError("Empty training data after preprocessing.")

    if debug_print_dataset_example and int(os.environ.get("RANK", "0")) == 0:
        ex = data[0]
        print("[dataset-debug] One full training example:")
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    return Dataset.from_list(data)


def build_reward_funcs(reward_names: List[str], args: argparse.Namespace):
    debug_state = {
        "proof_leaves": 0,
        "proof_steps": 0,
        "proof_intermediates": 0,
    }

    bleurt_scorer = None
    if args.proof_intermediates_reward_backend == "bleurt":
        if not args.bleurt_checkpoint:
            raise ValueError("--bleurt-checkpoint is required when --proof-intermediates-reward-backend=bleurt")
        try:
            from bleurt import score as bleurt_score
        except Exception as e:
            raise RuntimeError(
                "Failed to import BLEURT. Install it in the training env, e.g. pip install bleurt tensorflow"
            ) from e
        bleurt_scorer = bleurt_score.BleurtScorer(args.bleurt_checkpoint)
        if args.debug_reward and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[reward-debug][proof_intermediates] backend=bleurt checkpoint={args.bleurt_checkpoint} "
                f"threshold={args.bleurt_threshold} batch_size={args.bleurt_batch_size}"
            )

    def _trim_text(text: str) -> str:
        limit = max(0, int(args.debug_reward_max_text))
        if limit == 0:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "...<truncated>"

    def _should_print(name: str, sample_idx: int) -> bool:
        if not args.debug_reward:
            return False
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return False
        if debug_state[name] >= args.debug_reward_max_calls:
            return False
        if sample_idx >= args.debug_reward_max_samples:
            return False
        return True

    def _extract_pred_gold_steps(pred_text: str, gold_lines, gold_nodes):
        pred = graph_nodes_to_steps(parse_json_graph_text(pred_text))
        gold = graph_nodes_to_steps(gold_nodes)
        gold_text = json.dumps(gold_nodes, ensure_ascii=False)
        return pred, gold, gold_text

    def proof_leaves_reward(completions, graph_lines=None, graph_nodes=None, **kwargs) -> List[float]:
        rewards = []
        debug_state["proof_leaves"] += 1
        call_idx = debug_state["proof_leaves"]
        graph_lines = graph_lines or []
        graph_nodes = graph_nodes or []
        for sample_idx, c in enumerate(completions):
            pred_text = c[0].get("content", "")
            gold_lines_i = graph_lines[sample_idx] if sample_idx < len(graph_lines) else []
            gold_nodes_i = graph_nodes[sample_idx] if sample_idx < len(graph_nodes) else []
            pred, goldp, gold_text = _extract_pred_gold_steps(pred_text, gold_lines_i, gold_nodes_i)

            pred_leaves = set()
            for _, refs, _ in pred:
                pred_leaves.update(x for x in refs if x.startswith("sent"))
            gold_leaves = set()
            for _, refs, _ in goldp:
                gold_leaves.update(x for x in refs if x.startswith("sent"))
            f1 = f1_sets(pred_leaves, gold_leaves)
            reward_value = 2.0 * f1 - 1.0
            rewards.append(reward_value)
            if _should_print("proof_leaves", sample_idx):
                print("[reward-debug][proof_leaves]", {
                    "call_idx": call_idx,
                    "sample_idx": sample_idx,
                    "reward": reward_value,
                    "f1": f1,
                    "pred_leaves": sorted(pred_leaves),
                    "gold_leaves": sorted(gold_leaves),
                    "pred_text": _trim_text(pred_text),
                    "gold_text": _trim_text(gold_text),
                })
        return rewards

    def proof_steps_reward(completions, graph_lines=None, graph_nodes=None, **kwargs) -> List[float]:
        rewards = []
        debug_state["proof_steps"] += 1
        call_idx = debug_state["proof_steps"]
        graph_lines = graph_lines or []
        graph_nodes = graph_nodes or []
        for sample_idx, c in enumerate(completions):
            pred_text = c[0].get("content", "")
            gold_lines_i = graph_lines[sample_idx] if sample_idx < len(graph_lines) else []
            gold_nodes_i = graph_nodes[sample_idx] if sample_idx < len(graph_nodes) else []
            pred, goldp, gold_text = _extract_pred_gold_steps(pred_text, gold_lines_i, gold_nodes_i)

            pred_edges = set((lhs, tuple(sorted(refs))) for lhs, refs, _ in pred)
            gold_edges = set((lhs, tuple(sorted(refs))) for lhs, refs, _ in goldp)
            f1 = f1_sets(pred_edges, gold_edges)
            reward_value = 3.0 * f1 - 1.5
            rewards.append(reward_value)
            if _should_print("proof_steps", sample_idx):
                print("[reward-debug][proof_steps]", {
                    "call_idx": call_idx,
                    "sample_idx": sample_idx,
                    "reward": reward_value,
                    "f1": f1,
                    "pred_edges_count": len(pred_edges),
                    "gold_edges_count": len(gold_edges),
                    "pred_edges_sample": [list(x) for x in sorted(pred_edges)[:10]],
                    "gold_edges_sample": [list(x) for x in sorted(gold_edges)[:10]],
                    "pred_text": _trim_text(pred_text),
                    "gold_text": _trim_text(gold_text),
                })
        return rewards

    def _bleurt_node_score(pred_text: str, gold_text: str) -> float:
        assert bleurt_scorer is not None
        refs = [gold_text]
        cands = [pred_text]
        try:
            raw = bleurt_scorer.score(references=refs, candidates=cands, batch_size=args.bleurt_batch_size)[0]
        except TypeError:
            raw = bleurt_scorer.score(refs, cands, batch_size=args.bleurt_batch_size)[0]
        return float(raw)

    def proof_intermediates_reward(
        completions,
        graph_lines=None,
        graph_nodes=None,
        intermediate_texts=None,
        **kwargs,
    ) -> List[float]:
        rewards = []
        debug_state["proof_intermediates"] += 1
        call_idx = debug_state["proof_intermediates"]
        graph_lines = graph_lines or []
        graph_nodes = graph_nodes or []
        intermediate_texts = intermediate_texts or []

        for idx, c in enumerate(completions):
            pred_text = c[0].get("content", "")
            gold_lines_i = graph_lines[idx] if idx < len(graph_lines) else []
            gold_nodes_i = graph_nodes[idx] if idx < len(graph_nodes) else []
            pred, goldp, gold_text = _extract_pred_gold_steps(pred_text, gold_lines_i, gold_nodes_i)

            gold_by_id = {lhs: concl for lhs, _, concl in goldp if lhs.startswith("int")}
            if idx < len(intermediate_texts):
                node_texts = intermediate_texts[idx] or {}
                if node_texts:
                    gold_by_id = {k: str(v) for k, v in node_texts.items() if str(k).startswith("int")}
            pred_by_id = {lhs: concl for lhs, _, concl in pred if lhs.startswith("int")}

            if not gold_by_id:
                rewards.append(0.0)
                continue

            sims = []
            per_node = []
            for iid, gtxt in gold_by_id.items():
                ptxt = pred_by_id.get(iid, "")
                if args.proof_intermediates_reward_backend == "bleurt":
                    raw_bleurt = _bleurt_node_score(ptxt, gtxt)
                    node_score = 1.0 if raw_bleurt >= args.bleurt_threshold else 0.0
                else:
                    gtok = set(norm_text(gtxt).split())
                    ptok = set(norm_text(ptxt).split())
                    if not gtok and not ptok:
                        node_score = 1.0
                    elif not gtok or not ptok:
                        node_score = 0.0
                    else:
                        node_score = len(gtok & ptok) / len(gtok | ptok)
                    raw_bleurt = None
                sims.append(node_score)
                row = {
                    "id": iid,
                    "node_score": node_score,
                    "gold": _trim_text(gtxt),
                    "pred": _trim_text(ptxt),
                }
                if raw_bleurt is not None:
                    row["bleurt_raw"] = raw_bleurt
                    row["bleurt_threshold"] = args.bleurt_threshold
                per_node.append(row)

            score = sum(sims) / len(sims)
            reward_value = 2.0 * score - 1.0
            rewards.append(reward_value)
            if _should_print("proof_intermediates", idx):
                print("[reward-debug][proof_intermediates]", {
                    "call_idx": call_idx,
                    "sample_idx": idx,
                    "backend": args.proof_intermediates_reward_backend,
                    "reward": reward_value,
                    "score": score,
                    "per_node": per_node[:20],
                    "pred_text": _trim_text(pred_text),
                    "gold_text": _trim_text(gold_text),
                })
        return rewards

    reward_map = {
        "proof_leaves": proof_leaves_reward,
        "proof_steps": proof_steps_reward,
        "proof_intermediates": proof_intermediates_reward,
    }
    funcs = []
    for n in reward_names:
        key = n.strip().lower()
        if key not in reward_map:
            raise ValueError(f"Unknown reward: {key}")
        funcs.append(reward_map[key])
    return funcs


def resolve_paths(args: argparse.Namespace):
    rewards_tag = "_".join([x.strip() for x in args.rewards.split(",") if x.strip()])
    model_tag = args.model_name.split("/")[-1].replace(".", "_").replace("-", "_")
    mode_tag = "graph_json"
    default = (
        f"{model_tag}_entailment_task{args.task_id}_{mode_tag}_grpo_"
        f"{args.max_steps}_{rewards_tag}_{args.num_generations}_seed{args.seed}"
    )
    save_path = args.save_path or default
    output_dir = args.output_dir or save_path
    logging_dir = args.logging_dir or f"runs_{save_path}"
    return save_path, output_dir, logging_dir


def main() -> None:
    args = parse_args()
    reward_names = [x.strip() for x in args.rewards.split(",") if x.strip()]
    reward_funcs = build_reward_funcs(reward_names, args)
    save_path, output_dir, logging_dir = resolve_paths(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    maybe_set_gemma_chat_template(tokenizer, args.model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_checkpoint,
        is_trainable=True,
        adapter_name="default",
    )
    model.set_adapter("default")

    dataset = build_training_dataset(
        args.dataset_file,
        args.task_id,
        allow_system_role=not is_gemma_2_2b(args.model_name),
        debug_print_dataset_example=args.debug_print_dataset_example,
    )

    if args.gradient_accumulation_steps > 0:
        grad_acc = args.gradient_accumulation_steps
    else:
        denom = 4 * args.per_device_train_batch_size
        grad_acc = max(1, math.ceil(args.num_generations / max(1, denom)))

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[cfg] gradient_accumulation_steps={grad_acc}")

    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="tensorboard",
        logging_dir=logging_dir,
        output_dir=output_dir,
        fp16=True,
        bf16=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    if args.skip_save:
        if int(os.environ.get("RANK", "0")) == 0:
            print("[save] skip-save enabled: not saving adapter/tokenizer")
    else:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
