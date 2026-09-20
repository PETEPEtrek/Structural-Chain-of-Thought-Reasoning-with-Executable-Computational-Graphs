#!/usr/bin/env python3
"""Prepare balanced proof and graph data from official ProofWriter splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from proofwriter_data_utils import (
    balanced_reservoir_sample,
    build_master_row,
    filter_rows,
    write_json,
)


SCHEMA_VERSION = "proofwriter-master-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets_local/tasksource_proofwriter_default"),
        help="Local DatasetDict saved with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("proofwriter_data"),
        help="Directory for train, test, and combined master JSON files.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_split(
    rows: Iterable[Mapping[str, Any]],
    split: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    filtered = filter_rows(rows, proofwriter_config="depth-5")
    sampled = balanced_reservoir_sample(filtered, size=size, seed=seed)
    return [build_master_row(row, split=split) for row in sampled]


def distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(f"{row['answer']}/QDep={row['depth']}" for row in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("Split sizes must be non-negative")

    if not args.dataset_dir.exists():
        raise FileNotFoundError(
            f"Local ProofWriter dataset not found: {args.dataset_dir}. "
            "Download it on an internet-connected node before preparing data."
        )
    from datasets import load_from_disk

    splits = load_from_disk(str(args.dataset_dir))
    if "train" not in splits or "test" not in splits:
        raise ValueError("Local ProofWriter DatasetDict must contain train and test")
    train = prepare_split(splits["train"], "train", args.train_size, args.seed)
    test = prepare_split(splits["test"], "test", args.test_size, args.seed + 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "proofwriter_train.json"
    test_path = args.output_dir / "proofwriter_test.json"
    master_path = args.output_dir / "proofwriter_master.json"
    write_json(train_path, train)
    write_json(test_path, test)

    master = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": "tasksource/proofwriter",
            "dataset_config": "default",
            "row_config": "depth-5",
            "official_splits": ["train", "test"],
        },
        "sampling": {
            "seed": args.seed,
            "strata": ["answer", "QDep"],
            "answers": ["True", "False"],
            "depths": list(range(6)),
            "requested_sizes": {
                "train": args.train_size,
                "test": args.test_size,
            },
            "distribution": {
                "train": distribution(train),
                "test": distribution(test),
            },
        },
        "splits": {"train": train, "test": test},
    }
    master_path.write_text(
        json.dumps(master, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[done] train: {len(train)} -> {train_path}")
    print(f"[done] test: {len(test)} -> {test_path}")
    print(f"[done] master -> {master_path}")


if __name__ == "__main__":
    main()
