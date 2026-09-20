#!/usr/bin/env python3
"""Split gsm8k_new_ops.json into the file layout expected by the existing
SFT / GRPO pipelines.

Source layout for gsm8k_new_ops.json:
    {"train": [items...], "test": [items...]}

Output layout (mirrors the existing gsm8k_comp_graphs_* family):
    gsm8k_new_ops_comp_graphs_train.json      full train (5634)
    gsm8k_new_ops_comp_graphs_sft_train.json  first half of train (pre_grpo SFT)
    gsm8k_new_ops_comp_graphs_sft_grpo.json   second half of train (GRPO data)
    gsm8k_new_ops_comp_graphs_test.json

The split is deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="gsm8k_new_ops.json",
        help="Source JSON with {train, test} splits.",
    )
    parser.add_argument(
        "--out-prefix",
        default="gsm8k_new_ops_comp_graphs",
        help="Prefix for the produced split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to deterministically split train into sft/grpo halves.",
    )
    return parser.parse_args()


def write_json(path: Path, items: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"  wrote {path}  ({len(items)} items, {path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"Source file not found: {src}")

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not (isinstance(data, dict) and "train" in data and "test" in data):
        raise SystemExit(
            "Source JSON must be a dict with keys 'train' and 'test'."
        )

    train_items = list(data["train"])
    test_items = list(data["test"])

    rng = random.Random(args.seed)
    indices = list(range(len(train_items)))
    rng.shuffle(indices)
    half = len(indices) // 2
    sft_train = [train_items[i] for i in indices[:half]]
    sft_grpo = [train_items[i] for i in indices[half:]]

    out_prefix = args.out_prefix
    out_dir = Path(out_prefix).parent if "/" in out_prefix else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source: {src}  train={len(train_items)}  test={len(test_items)}")
    print(
        f"sft_train={len(sft_train)}  sft_grpo={len(sft_grpo)}  seed={args.seed}"
    )

    pairs = [
        (f"{out_prefix}_train.json", train_items),
        (f"{out_prefix}_sft_train.json", sft_train),
        (f"{out_prefix}_sft_grpo.json", sft_grpo),
        (f"{out_prefix}_test.json", test_items),
    ]
    for path_str, items in pairs:
        write_json(Path(path_str), items)

    print("\nDone.")


if __name__ == "__main__":
    main()
