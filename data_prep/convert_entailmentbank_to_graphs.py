#!/usr/bin/env python3
"""
Convert EntailmentBank jsonl into a graph dataset, and optionally evaluate
predicted graphs against gold graphs.
"""

import argparse
import json
import re
from pathlib import Path


STEP_SEP_RE = re.compile(r";\s*")
SENT_RE = re.compile(r"\b(sent\d+)\b")
LHS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_convert = sub.add_parser("convert", help="Convert EntailmentBank jsonl to graph datasets.")
    p_convert.add_argument("--input-jsonl", required=True, type=Path)
    p_convert.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Output with computational_graph (JSON nodes).",
    )
    p_convert.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, convert only first N rows (debug).",
    )

    p_eval = sub.add_parser("eval", help="Evaluate predicted graph outputs.")
    p_eval.add_argument("--gold-json", required=True, type=Path)
    p_eval.add_argument(
        "--predictions-jsonl",
        required=True,
        type=Path,
        help=(
            "JSONL predictions with at least id + one of: "
            "computational_graph_transformed, model_graph, model_response."
        ),
    )
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.error("A subcommand is required: convert or eval")
    return args


def parse_context_triples(example: dict) -> dict:
    triples = ((example.get("meta") or {}).get("triples")) or {}
    if triples:
        return dict(triples)

    context = example.get("context") or ""
    out = {}
    # Parse "sent1: ... sent2: ..." format
    matches = list(re.finditer(r"(sent\d+)\s*:\s*", context))
    for i, m in enumerate(matches):
        sid = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        out[sid] = context[start:end].strip()
    return out


def parse_proof_steps(proof: str, hypothesis_text: str) -> list:
    if not proof:
        return []

    steps = []
    for raw_step in STEP_SEP_RE.split(proof.strip()):
        step = raw_step.strip()
        if not step or "->" not in step:
            continue
        left, right = step.split("->", 1)
        inputs = [x.strip() for x in left.split("&") if x.strip()]
        right = right.strip()

        if ":" in right:
            out_id, out_text = right.split(":", 1)
            out_id = out_id.strip()
            out_text = out_text.strip()
        else:
            out_id = right
            out_text = ""

        if out_id == "hypothesis" and not out_text:
            out_text = hypothesis_text or ""

        steps.append(
            {
                "inputs": inputs,
                "out_id": out_id,
                "out_text": out_text,
            }
        )
    return steps


def build_graph_nodes(context_triples: dict, steps: list) -> list:
    referenced_sents = []
    seen = set()
    for st in steps:
        for inp in st["inputs"]:
            if inp.startswith("sent") and inp not in seen:
                seen.add(inp)
                referenced_sents.append(inp)

    referenced_sents.sort(key=lambda x: int(x.replace("sent", "")))
    nodes = []
    for sid in referenced_sents:
        nodes.append(
            {
                "id": sid,
                "type": "const",
                "value": context_triples.get(sid, ""),
            }
        )

    for st in steps:
        nodes.append(
            {
                "id": st["out_id"],
                "type": "op",
                "op": "entail_and",
                "inputs": st["inputs"],
                "value": st["out_text"],
            }
        )
    return nodes


def build_assignment_lines(steps: list, hypothesis_text: str) -> list:
    lines = []
    for st in steps:
        rhs = " & ".join(st["inputs"])
        out_text = (st.get("out_text") or "").strip()
        if st.get("out_id") == "hypothesis" and not out_text:
            out_text = (hypothesis_text or "").strip()
        lines.append(f"{st['out_id']}={rhs} :: {out_text}")
    return lines


def normalize_record(example: dict) -> dict:
    hypothesis = example.get("hypothesis", "")
    proof = example.get("proof", "")
    context_triples = parse_context_triples(example)
    steps = parse_proof_steps(proof, hypothesis_text=hypothesis)
    graph_nodes = build_graph_nodes(context_triples, steps)
    assignment_lines = build_assignment_lines(steps, hypothesis_text=hypothesis)

    return {
        "id": example.get("id"),
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),
        "hypothesis": hypothesis,
        "context": example.get("context", ""),
        "proof": proof,
        "computational_graph": graph_nodes,
        "computational_graph_transformed": assignment_lines,
    }


def convert(input_jsonl: Path, output_json: Path, limit: int) -> None:
    lines = input_jsonl.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[:limit]

    records = []
    skipped_no_proof = 0
    for ln in lines:
        ex = json.loads(ln)
        if not ex.get("proof"):
            skipped_no_proof += 1
            continue
        records.append(normalize_record(ex))

    output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Converted records: {len(records)}")
    print(f"Skipped (empty proof): {skipped_no_proof}")
    print(f"Wrote: {output_json}")


def parse_entail_line(line: str):
    m = LHS_RE.match(line)
    if not m:
        return None
    lhs = m.group(1).strip()
    rhs = m.group(2).strip()
    # Support both:
    # 1) lhs=entail_and(a AND b)
    # 2) lhs=a & b
    inside = rhs
    if "::" in inside:
        inside = inside.split("::", 1)[0].strip()
    if inside.startswith("entail_and(") and inside.endswith(")"):
        inside = inside[len("entail_and(") : -1].strip()
    if not inside:
        return None
    refs = [x.strip() for x in re.split(r"\s*(?:AND|&)\s*", inside) if x.strip()]
    if not refs:
        return None
    return lhs, tuple(sorted(refs))


def gold_edges(gold_rec: dict) -> set:
    edges = set()
    for node in gold_rec.get("computational_graph", []):
        if node.get("type") != "op":
            continue
        refs = tuple(sorted([str(x).strip() for x in node.get("inputs", []) if str(x).strip()]))
        if refs:
            edges.add((node.get("id"), refs))
    return edges


def pred_edges(pred_rec: dict) -> set:
    # priority 1: explicit assignment lines
    lines = pred_rec.get("computational_graph_transformed")
    if isinstance(lines, list):
        out = set()
        for ln in lines:
            if not isinstance(ln, str):
                continue
            p = parse_entail_line(ln.strip())
            if p is not None:
                out.add(p)
        return out

    # priority 2: model_graph as a list of assignment lines
    mg = pred_rec.get("model_graph")
    if isinstance(mg, list) and mg and isinstance(mg[0], str):
        out = set()
        for ln in mg:
            p = parse_entail_line(str(ln).strip())
            if p is not None:
                out.add(p)
        return out

    # priority 3: parse model_response line by line
    text = pred_rec.get("model_response")
    if isinstance(text, str):
        out = set()
        for raw in text.splitlines():
            p = parse_entail_line(raw.strip())
            if p is not None:
                out.add(p)
        return out
    return set()


def evaluate(gold_json: Path, predictions_jsonl: Path) -> None:
    gold = json.loads(gold_json.read_text(encoding="utf-8"))
    gold_by_id = {str(x.get("id")): x for x in gold}

    pred_lines = predictions_jsonl.read_text(encoding="utf-8").splitlines()
    pred_by_id = {}
    for ln in pred_lines:
        row = json.loads(ln)
        rid = str(row.get("id", ""))
        if rid:
            pred_by_id[rid] = row

    total = 0
    exact = 0
    tp = 0
    fp = 0
    fn = 0
    missing_pred = 0

    for rid, grec in gold_by_id.items():
        total += 1
        gedges = gold_edges(grec)
        prec = pred_by_id.get(rid)
        if prec is None:
            missing_pred += 1
            fn += len(gedges)
            continue
        pedges = pred_edges(prec)
        inter = gedges & pedges
        tp += len(inter)
        fp += len(pedges - gedges)
        fn += len(gedges - pedges)
        if gedges == pedges and len(gedges) > 0:
            exact += 1

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"Gold rows: {total}")
    print(f"Missing predictions by id: {missing_pred}")
    print(f"Exact graph match: {exact}/{total} = {exact/total if total else 0.0:.4f}")
    print(f"Edge precision: {precision:.4f}")
    print(f"Edge recall:    {recall:.4f}")
    print(f"Edge F1:        {f1:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "convert":
        convert(
            input_jsonl=args.input_jsonl,
            output_json=args.output_json,
            limit=args.limit,
        )
        return
    if args.command == "eval":
        evaluate(gold_json=args.gold_json, predictions_jsonl=args.predictions_jsonl)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
