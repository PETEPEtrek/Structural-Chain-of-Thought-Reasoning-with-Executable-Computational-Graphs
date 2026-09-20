#!/usr/bin/env python3
"""
Prepare EntailmentBank baseline training JSON from original jsonl splits.

Output keeps original fields (question, answer, hypothesis, context, proof,
full_text_proof) without graph conversion.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Original EntailmentBank jsonl (train or test).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON list for baseline SFT/GRPO scripts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, keep only first N rows (debug).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_row(row: dict) -> Optional[dict]:
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    proof = (row.get("proof") or "").strip()
    if not question or not answer or not proof:
        return None
    return {
        "id": row.get("id", ""),
        "question": question,
        "answer": answer,
        "hypothesis": (row.get("hypothesis") or "").strip(),
        "context": (row.get("context") or "").strip(),
        "proof": (row.get("proof") or "").strip(),
        "full_text_proof": (row.get("full_text_proof") or "").strip(),
    }


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"Missing input jsonl: {args.input_jsonl}")

    raw_rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        raw_rows = raw_rows[: args.limit]

    rows = []
    skipped = 0
    for row in raw_rows:
        normalized = normalize_row(row)
        if normalized is None:
            skipped += 1
            continue
        rows.append(normalized)

    if not rows:
        raise ValueError(f"No valid rows produced from {args.input_jsonl}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] wrote {len(rows)} rows to {args.output_json}"
        + (f" (skipped {skipped})" if skipped else "")
    )


if __name__ == "__main__":
    main()
