#!/usr/bin/env python3
"""Convert ProofWriter master rows into records for every training approach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from proofwriter_prompts import (
    answer_target,
    build_user_input,
    cot_answer_target,
    default_grpo_full_target,
    default_grpo_solution_target,
    get_system_prompt,
    graph_target,
    normalize_boolean,
    to_entailment_graph,
)

METHODS = (
    "answer_sft",
    "cot_answer_sft",
    "default_grpo",
    "graph_sft",
    "graph_grpo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("proofwriter_data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("proofwriter_training_data"),
        help="Root containing one records directory per method.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def all_gold_graphs(row: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    """Return canonical first, followed by every distinct proof alternative."""
    candidates = [row.get("canonical_graph")]
    candidates.extend(
        proof.get("graph")
        for proof in (row.get("gold_proofs") or [])
        if isinstance(proof, dict)
    )
    result: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for graph in candidates:
        if not isinstance(graph, (dict, list)):
            continue
        compact_graph = to_entailment_graph(graph)
        signature = json.dumps(compact_graph, ensure_ascii=False, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            result.append(compact_graph)
    if not result:
        raise ValueError(f"Row {row.get('id', '<unknown>')} has no gold graph")
    return result


def build_record(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    user = build_user_input(row)
    prompt = [
        {"role": "system", "content": get_system_prompt(method)},
        {"role": "user", "content": user},
    ]
    base = {
        "id": str(row.get("id", "")),
        "method": method,
        "prompt": prompt,
        "answer": normalize_boolean(row.get("answer")),
    }
    if method in {"answer_sft", "cot_answer_sft", "graph_sft"}:
        target_fn = {
            "answer_sft": answer_target,
            "cot_answer_sft": cot_answer_target,
            "graph_sft": graph_target,
        }[method]
        target = target_fn(row)
        base.update(
            {
                "messages": [*prompt, {"role": "assistant", "content": target}],
                "target": target,
                "completion": [{"role": "assistant", "content": target}],
            }
        )
    else:
        base.update(
            {
                "gold_graphs": all_gold_graphs(row),
                "canonical_graph": row.get("canonical_graph"),
                "canonical_proof": row.get("canonical_proof", ""),
                "facts": row.get("facts") or {},
                "rules": row.get("rules") or {},
                "question": row.get("question", ""),
                "proved_literal": row.get("proved_literal", ""),
            }
        )
        if method == "default_grpo":
            # Gold bodies for format reference / offline checks.
            base["solution_target"] = default_grpo_solution_target(row)
            base["full_target"] = default_grpo_full_target(row)
    return base


def build_records(
    rows: Iterable[Mapping[str, Any]], method: str
) -> list[dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return [build_record(row, method) for row in rows]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {"schema_version": "proofwriter-training-v1", "files": {}}
    for split in ("train", "test"):
        source = args.input_dir / f"proofwriter_{split}.json"
        rows = load_rows(source)
        for method in METHODS:
            records = build_records(rows, method)
            destination = args.output_dir / method / "records" / f"{split}.json"
            write_records(destination, records)
            manifest["files"][f"{method}/{split}"] = {
                "path": str(destination),
                "count": len(records),
            }
            print(f"[done] {method}/{split}: {len(records)} -> {destination}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
