#!/usr/bin/env python3
"""GRPO for ProofWriter full proof graphs.

Reward ranges (per completion):
  graph_style/schema: -1 or +2
  object/node overlap: -1 .. +2
  connection/edge overlap: -1.5 .. +3
  max-over-gold graph match: -1 .. +3
  boolean answer: -1.5, 0, or +3
  proof faithfulness: -1.5 or +3

All structural rewards maximize over every gold proof alternative. Intermediate
node ids do not participate in matching; indexed source-fact ids remain meaningful.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import activations

from prepare_proofwriter_training_data import build_record
from proofwriter_data_utils import literal_key
from proofwriter_default_grpo_script import _content
from proofwriter_prompts import maybe_set_gemma_chat_template, normalize_boolean

# transformers>=4.52 removed this symbol, while AutoAWQ imported by PEFT
# still expects it.
activations.PytorchGELUTanh = activations.GELUTanh

DEFAULT_NUM_GENERATIONS = 16
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", str(value).casefold())).strip()


def validate_graph_schema(graph: Any) -> bool:
    """Validate an EntailmentBank-style ordered JSON-array proof graph."""
    if not isinstance(graph, list) or not graph:
        return False
    seen: set[str] = set()
    for node in graph:
        if not isinstance(node, dict):
            return False
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            return False
        if not isinstance(node.get("value"), str) or not node["value"].strip():
            return False
        if node_type == "const":
            if set(node) != {"id", "type", "value"}:
                return False
        elif node_type == "op":
            if set(node) != {"id", "type", "op", "rule", "inputs", "value"}:
                return False
            if node.get("op") != "apply_rule":
                return False
            if not isinstance(node.get("rule"), str) or not node["rule"]:
                return False
            inputs = node.get("inputs")
            if (
                not isinstance(inputs, list)
                or not inputs
                or any(not isinstance(item, str) or item not in seen for item in inputs)
            ):
                return False
        else:
            return False
        seen.add(node_id)
    return True


GRAPH_RESPONSE_RE = re.compile(
    r"\A\s*(?P<graph>\[.*\])\s*\nANSWER: (?P<answer>True|False)\s*\Z",
    flags=re.DOTALL,
)


def parse_graph_response(
    text: Any,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse an exact JSON graph plus boolean ANSWER line."""
    try:
        match = GRAPH_RESPONSE_RE.fullmatch(str(text or ""))
        if not match:
            return None, None
        graph = json.loads(match.group("graph"))
        if not validate_graph_schema(graph):
            return None, match.group("answer")
        return graph, match.group("answer")
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return None, None


def node_signature(node: Mapping[str, Any]) -> tuple[str, ...]:
    if node.get("type") == "op":
        return (
            "op",
            str(node.get("rule", "")).casefold(),
            norm_text(node.get("value", "")),
        )
    return (
        "const",
        str(node.get("id", "")).casefold(),
        norm_text(node.get("value", "")),
    )


def canonical_graph_signature(
    graph: list[Mapping[str, Any]],
) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[Any, ...]], tuple[str, ...]]:
    by_id = {node["id"]: node_signature(node) for node in graph}
    node_sigs = frozenset(by_id.values())
    edge_sigs = frozenset(
        ("input", by_id[source], by_id[node["id"]], order)
        for node in graph
        if node["type"] == "op"
        for order, source in enumerate(node["inputs"])
    )
    return node_sigs, edge_sigs, by_id[graph[-1]["id"]]


def f1_sets(predicted: set | frozenset, gold: set | frozenset) -> float:
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


def _sanitize_graph_node(node: Any) -> Any:
    """Drop null-padded keys inserted by datasets.Dataset feature alignment."""
    if not isinstance(node, dict):
        return node
    return {key: value for key, value in node.items() if value is not None}


def _sanitize_graph(graph: Any) -> Any:
    if not isinstance(graph, list):
        return graph
    return [_sanitize_graph_node(node) for node in graph]


def _valid_gold_graphs(graphs: Any) -> list[list[dict[str, Any]]]:
    if isinstance(graphs, list) and graphs and all(
        isinstance(node, dict) for node in graphs
    ):
        graphs = [graphs]
    if not isinstance(graphs, list):
        return []
    valid = []
    for graph in graphs:
        cleaned = _sanitize_graph(graph)
        if validate_graph_schema(cleaned):
            valid.append(cleaned)
    return valid


def _batch_gold(gold_graphs: Any, size: int) -> list[Any]:
    if not isinstance(gold_graphs, list):
        return [[] for _ in range(size)]
    # TRL supplies one list-of-graphs per completion. A direct unit test may
    # supply one shared list of graph alternatives.
    sanitized = [_sanitize_graph(item) for item in gold_graphs]
    if sanitized and all(validate_graph_schema(item) for item in sanitized):
        return [sanitized for _ in range(size)]
    return sanitized


def graph_style_reward(completions, **kwargs) -> list[float]:
    return [
        2.0 if parse_graph_response(_content(completion))[0] is not None else -1.0
        for completion in completions
    ]


def _overlap_rewards(completions, gold_graphs, component: int, scale: float, floor: float):
    scores = []
    batch_gold = _batch_gold(gold_graphs, len(completions))
    for index, completion in enumerate(completions):
        predicted, _ = parse_graph_response(_content(completion))
        alternatives = _valid_gold_graphs(
            batch_gold[index] if index < len(batch_gold) else []
        )
        if predicted is None or not alternatives:
            scores.append(floor)
            continue
        pred_sig = canonical_graph_signature(predicted)[component]
        best = max(
            f1_sets(pred_sig, canonical_graph_signature(gold)[component])
            for gold in alternatives
        )
        scores.append(floor + (scale - floor) * best)
    return scores


def object_overlap_reward(completions, gold_graphs, **kwargs) -> list[float]:
    return _overlap_rewards(completions, gold_graphs, 0, 2.0, -1.0)


def connection_overlap_reward(completions, gold_graphs, **kwargs) -> list[float]:
    return _overlap_rewards(completions, gold_graphs, 1, 3.0, -1.5)


def graph_match_reward(completions, gold_graphs, **kwargs) -> list[float]:
    """Combined structural F1, maximized jointly over alternative proofs."""
    scores = []
    batch_gold = _batch_gold(gold_graphs, len(completions))
    for index, completion in enumerate(completions):
        predicted, _ = parse_graph_response(_content(completion))
        alternatives = _valid_gold_graphs(
            batch_gold[index] if index < len(batch_gold) else []
        )
        if predicted is None or not alternatives:
            scores.append(-1.0)
            continue
        pred_nodes, pred_edges, _ = canonical_graph_signature(predicted)
        best = max(
            (
                f1_sets(pred_nodes, canonical_graph_signature(gold)[0])
                + f1_sets(pred_edges, canonical_graph_signature(gold)[1])
            )
            / 2.0
            for gold in alternatives
        )
        scores.append(-1.0 + 4.0 * best)
    return scores


def boolean_answer_reward(completions, answer, **kwargs) -> list[float]:
    """Score the explicit boolean in the final ANSWER line."""
    scores = []
    for completion, gold_answer in zip(completions, answer):
        _, predicted_answer = parse_graph_response(_content(completion))
        if predicted_answer is None:
            scores.append(0.0)
            continue
        try:
            correct = predicted_answer == normalize_boolean(gold_answer)
        except ValueError:
            correct = False
        scores.append(3.0 if correct else -1.5)
    return scores


def _applications(graph: list[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    by_id = {node["id"]: node for node in graph}
    applications: set[tuple[Any, ...]] = set()
    for node in graph:
        if node["type"] != "op":
            continue
        applications.add(
            (
                str(node["rule"]).casefold(),
                tuple(node_signature(by_id[source]) for source in node["inputs"]),
                node_signature(node),
            )
        )
    return applications


def _is_complete_rooted_proof(graph: list[Mapping[str, Any]]) -> bool:
    by_id = {node["id"]: node for node in graph}
    ancestors = {graph[-1]["id"]}
    stack = [graph[-1]["id"]]
    while stack:
        current = stack.pop()
        node = by_id[current]
        for source in node.get("inputs", []):
            if source not in ancestors:
                ancestors.add(source)
                stack.append(source)
    return ancestors == set(by_id)


def is_faithful_graph(
    predicted: Any, gold_graphs: Iterable[list[Mapping[str, Any]]]
) -> bool:
    """Execute structural proof obligations against one complete gold proof."""
    if (
        not validate_graph_schema(predicted) or not _is_complete_rooted_proof(predicted)
    ):
        return False
    pred_apps = _applications(predicted)
    pred_nodes, _, pred_root = canonical_graph_signature(predicted)
    for gold in _valid_gold_graphs(list(gold_graphs)):
        gold_apps = _applications(gold)
        gold_nodes, _, gold_root = canonical_graph_signature(gold)
        if pred_root != gold_root:
            continue
        # Every fact and every rule execution must be supported by this same
        # proof alternative; this prevents mixing incompatible gold proofs.
        if pred_apps <= gold_apps and pred_nodes <= gold_nodes:
            return True
    return False


def faithfulness_reward(
    completions, gold_graphs, question, **kwargs
) -> list[float]:
    """Require a supported proof and agreement between ANSWER and its last node."""
    scores = []
    batch_gold = _batch_gold(gold_graphs, len(completions))
    for index, (completion, query) in enumerate(zip(completions, question)):
        predicted, predicted_answer = parse_graph_response(_content(completion))
        alternatives = batch_gold[index] if index < len(batch_gold) else []
        consistent = False
        if predicted is not None and predicted_answer is not None:
            query_base, query_negated = literal_key(query)
            root_base, root_negated = literal_key(predicted[-1]["value"])
            if query_base == root_base:
                implied_answer = "True" if query_negated == root_negated else "False"
                consistent = predicted_answer == implied_answer
        faithful = is_faithful_graph(predicted, alternatives)
        scores.append(3.0 if faithful and consistent else -1.5)
    return scores


def load_training_records(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    records = []
    for row in rows:
        record = (
            row
            if row.get("method") == "graph_grpo" and row.get("prompt")
            else build_record(row, "graph_grpo")
        )
        records.append(
            {
                "prompt": record["prompt"],
                "answer": record["answer"],
                "question": record["question"],
                "proved_literal": record["proved_literal"],
                "gold_graphs": record["gold_graphs"],
            }
        )
    if not records:
        raise ValueError("No graph GRPO records produced")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-model-dir", default=None)
    return parser.parse_args()


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    args = parse_args()
    set_seed(args.seed)
    dataset = Dataset.from_list(load_training_records(args.dataset_json)).shuffle(
        seed=args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    maybe_set_gemma_chat_template(tokenizer, args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if args.sft_checkpoint:
        model = PeftModel.from_pretrained(base, args.sft_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=LORA_TARGET_MODULES,
                inference_mode=False,
            ),
        )
    name = f"{Path(args.model_path).name}_proofwriter_graph_grpo_{args.num_generations}_seed{args.seed}"
    output_dir = args.output_dir or f"outputs_{name}"
    config = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_strategy="no",
        max_grad_norm=0.1,
        report_to="tensorboard",
        output_dir=output_dir,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        fp16=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            graph_style_reward,
            object_overlap_reward,
            connection_overlap_reward,
            graph_match_reward,
            boolean_answer_reward,
            faithfulness_reward,
        ],
        args=config,
        train_dataset=dataset,
    )
    print(f"train_examples: {len(dataset)}")
    trainer.train()
    save_model_dir = args.save_model_dir or name
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    main()
