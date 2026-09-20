#!/usr/bin/env python3
"""Utilities for preparing balanced ProofWriter proof/graph datasets."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence


_TOKEN_RE = re.compile(r"->|[()]|[A-Za-z_][A-Za-z_0-9-]*")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Proof:
    """A fact leaf or an application of one ProofWriter rule."""

    kind: str
    identifier: str
    premises: tuple["Proof", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind}
        if self.kind == "fact":
            result["fact_id"] = self.identifier
        else:
            result["rule_id"] = self.identifier
            result["premises"] = [premise.to_dict() for premise in self.premises]
        return result

    @property
    def rule_count(self) -> int:
        return (1 if self.kind == "apply_rule" else 0) + sum(
            premise.rule_count for premise in self.premises
        )

    @property
    def node_count(self) -> int:
        return 1 + sum(premise.node_count for premise in self.premises)


@dataclass(frozen=True)
class _Group:
    items: tuple[Proof, ...]


class ProofParseError(ValueError):
    """Raised when a ProofWriter proof expression is malformed."""


def _strip_outer_parentheses(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wraps_all = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ProofParseError(f"Unbalanced parentheses: {text!r}")
                if depth == 0 and index != len(text) - 1:
                    wraps_all = False
                    break
        if depth != 0:
            raise ProofParseError(f"Unbalanced parentheses: {text!r}")
        if not wraps_all:
            break
        text = text[1:-1].strip()
    return text


def split_proof_alternatives(text: str) -> list[str]:
    """Split an ``OR`` list without splitting inside nested proof expressions."""

    text = _strip_outer_parentheses(text.strip())
    alternatives: list[str] = []
    start = 0
    depth = 0
    for match in re.finditer(r"\(|\)|\bOR\b", text):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ProofParseError(f"Unbalanced parentheses: {text!r}")
        elif depth == 0:
            alternative = text[start : match.start()].strip()
            if not alternative:
                raise ProofParseError(f"Empty proof alternative: {text!r}")
            alternatives.append(alternative)
            start = match.end()
    if depth != 0:
        raise ProofParseError(f"Unbalanced parentheses: {text!r}")
    last = text[start:].strip()
    if last:
        alternatives.append(last)
    if not alternatives:
        raise ProofParseError("Proof alternative list is empty")
    return alternatives


class _ProofParser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = _TOKEN_RE.findall(expression)
        compact = _TOKEN_RE.sub("", expression)
        if compact.strip():
            raise ProofParseError(
                f"Unexpected characters {compact!r} in proof {expression!r}"
            )
        self.position = 0

    def parse(self) -> Proof:
        if not self.tokens:
            raise ProofParseError("Proof expression is empty")
        left: list[Proof] = []
        while self._peek() not in ("->", None):
            item = self._item()
            left.extend(item.items if isinstance(item, _Group) else (item,))
        if not left:
            raise ProofParseError(f"Proof has no premises: {self.expression!r}")
        if self._peek() == "->":
            self.position += 1
            rule_id = self._peek()
            if rule_id in (None, "(", ")", "->"):
                raise ProofParseError(f"Missing rule identifier in {self.expression!r}")
            self.position += 1
            result: Proof | _Group = Proof("apply_rule", rule_id, tuple(left))
        elif len(left) == 1:
            result = left[0]
        else:
            result = _Group(tuple(left))
        if self.position != len(self.tokens):
            raise ProofParseError(
                f"Unexpected token {self.tokens[self.position]!r} in {self.expression!r}"
            )
        return self._as_single_proof(result)

    def _item(self) -> Proof | _Group:
        token = self._peek()
        if token == "(":
            return self._parenthesized()
        if token in (None, ")", "->"):
            raise ProofParseError(
                f"Expected proof atom at token {self.position} in {self.expression!r}"
            )
        self.position += 1
        return Proof("fact", token)

    def _parenthesized(self) -> Proof | _Group:
        self._take("(")
        left: list[Proof] = []
        while self._peek() not in ("->", ")", None):
            item = self._item()
            left.extend(item.items if isinstance(item, _Group) else (item,))
        if not left:
            raise ProofParseError(f"Empty group in {self.expression!r}")

        if self._peek() == "->":
            self.position += 1
            rule_id = self._peek()
            if rule_id in (None, "(", ")", "->"):
                raise ProofParseError(f"Missing rule identifier in {self.expression!r}")
            self.position += 1
            self._take(")")
            return Proof("apply_rule", rule_id, tuple(left))

        self._take(")")
        if len(left) == 1:
            return left[0]
        return _Group(tuple(left))

    def _as_single_proof(self, item: Proof | _Group) -> Proof:
        if isinstance(item, _Group):
            raise ProofParseError(
                f"Premise group has no applied rule in {self.expression!r}"
            )
        return item

    def _peek(self) -> Optional[str]:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def _take(self, expected: str) -> None:
        actual = self._peek()
        if actual != expected:
            raise ProofParseError(
                f"Expected {expected!r}, got {actual!r} in {self.expression!r}"
            )
        self.position += 1


def parse_proof_expression(expression: str) -> Proof:
    """Parse nested ``((premises) -> ruleN)`` ProofWriter syntax."""

    return _ProofParser(expression.strip()).parse()


def parse_all_proofs(all_proofs: str) -> list[dict[str, Any]]:
    """Parse every assertion and every alternative from an ``allProofs`` field."""

    entries: list[dict[str, Any]] = []
    sections = list(re.finditer(r"@(\d+)\s*:", all_proofs))
    for index, section in enumerate(sections):
        depth = int(section.group(1))
        end = sections[index + 1].start() if index + 1 < len(sections) else len(all_proofs)
        body = all_proofs[section.end() : end]
        for match in re.finditer(r"([^[]+?)\s*\[([^\]]+)\]", body):
            assertion = match.group(1).strip()
            alternatives_text = match.group(2).strip()
            if assertion.endswith("."):
                assertion = assertion[:-1].rstrip() + "."
            alternatives = []
            for raw in split_proof_alternatives(alternatives_text):
                alternatives.append(
                    {"expression": raw, "tree": parse_proof_expression(raw)}
                )
            entries.append(
                {"depth": depth, "assertion": assertion, "proofs": alternatives}
            )
    if all_proofs.strip() and not entries:
        raise ProofParseError("No assertions could be parsed from allProofs")
    return entries


def normalize_sentence(sentence: str) -> str:
    sentence = _SPACE_RE.sub(" ", sentence.strip()).rstrip(".?!").strip()
    return sentence.casefold()


def _third_person(verb: str) -> str:
    lower = verb.casefold()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        return verb + "es"
    return verb + "s"


def literal_key(sentence: str) -> tuple[str, bool]:
    """Return a normalized positive form and whether the literal is negated."""

    normalized = normalize_sentence(sentence)
    for marker in (" is not ", " are not ", " was not ", " were not "):
        if marker in normalized:
            return normalized.replace(marker, marker.replace(" not", ""), 1), True
    match = re.match(r"^(.*?)\s+does not\s+([a-z-]+)(\s+.+)$", normalized)
    if match:
        positive = f"{match.group(1)} {_third_person(match.group(2))}{match.group(3)}"
        return positive, True
    return normalized, False


def explicit_complement(
    sentence: str, available_assertions: Sequence[str] = ()
) -> str:
    """Find or construct the explicit natural-language complement of a literal."""

    base, negated = literal_key(sentence)
    for assertion in available_assertions:
        other_base, other_negated = literal_key(assertion)
        if base == other_base and negated != other_negated:
            return assertion.strip()

    clean = _SPACE_RE.sub(" ", sentence.strip())
    suffix = "." if clean.endswith(".") else ""
    core = clean.rstrip(".?!")
    if negated:
        core = re.sub(r"\b(is|are|was|were) not\b", r"\1", core, count=1)
        relation = re.match(r"^(.*?)\s+does not\s+([A-Za-z-]+)(\s+.+)$", core)
        if relation:
            core = (
                f"{relation.group(1)} {_third_person(relation.group(2))}"
                f"{relation.group(3)}"
            )
    else:
        copula = re.search(r"\b(is|are|was|were)\b", core)
        if copula:
            core = core[: copula.end()] + " not" + core[copula.end() :]
        else:
            words = core.split()
            if len(words) < 3:
                raise ValueError(f"Cannot construct explicit complement for {sentence!r}")
            verb = words[1]
            if verb.endswith("ies"):
                lemma = verb[:-3] + "y"
            elif verb.endswith("es") and verb[:-2].endswith(("s", "x", "z", "ch", "sh", "o")):
                lemma = verb[:-2]
            elif verb.endswith("s"):
                lemma = verb[:-1]
            else:
                lemma = verb
            core = " ".join([words[0], "does", "not", lemma, *words[2:]])
    return core + suffix


def select_gold_proofs(
    all_proofs: str, question: str, answer: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the proved literal and all proof alternatives supporting the answer."""

    parsed = parse_all_proofs(all_proofs)
    assertions = [entry["assertion"] for entry in parsed]
    target = question.strip()
    if answer.casefold() == "false":
        target = explicit_complement(target, assertions)
    elif answer.casefold() != "true":
        raise ValueError(f"Only True/False answers have proofs, got {answer!r}")

    target_key = literal_key(target)
    matching = [entry for entry in parsed if literal_key(entry["assertion"]) == target_key]
    alternatives: list[dict[str, Any]] = []
    for entry in matching:
        for proof in entry["proofs"]:
            alternatives.append(
                {
                    "depth": entry["depth"],
                    "assertion": entry["assertion"],
                    "expression": proof["expression"],
                    "tree": proof["tree"],
                }
            )
    if not alternatives:
        raise ValueError(
            f"No proof for target literal {target!r} (question={question!r}, answer={answer})"
        )
    return matching[0]["assertion"], alternatives, parsed


def proof_signature(proof: Proof) -> tuple[Any, ...]:
    if proof.kind == "fact":
        return ("fact", proof.identifier)
    return (
        "apply_rule",
        proof.identifier,
        tuple(proof_signature(premise) for premise in proof.premises),
    )


def canonical_proof(proofs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not proofs:
        raise ValueError("Cannot choose a canonical proof from an empty list")
    return min(
        proofs,
        key=lambda item: (
            item.get("_tree", item["tree"]).rule_count,
            item.get("_tree", item["tree"]).node_count,
            _SPACE_RE.sub("", item["expression"]).casefold(),
        ),
    )


def index_theory(
    theory: str, n_facts: Optional[int] = None, n_rules: Optional[int] = None
) -> tuple[dict[str, str], dict[str, str]]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(theory.strip()) if part.strip()]
    if n_facts is None:
        n_facts = next(
            (index for index, text in enumerate(sentences) if text.casefold().startswith(("if ", "all "))),
            len(sentences),
        )
    facts = {
        f"triple{index + 1}": text for index, text in enumerate(sentences[:n_facts])
    }
    rule_texts = sentences[n_facts : n_facts + n_rules if n_rules is not None else None]
    rules = {f"rule{index + 1}": text for index, text in enumerate(rule_texts)}
    return facts, rules


def _conclusion_lookup(parsed_all: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], str]:
    lookup: dict[tuple[Any, ...], str] = {}
    for entry in parsed_all:
        for alternative in entry["proofs"]:
            lookup.setdefault(proof_signature(alternative["tree"]), entry["assertion"])
    return lookup


def proof_to_graph(
    proof: Proof,
    conclusion: str,
    facts: Mapping[str, str],
    rules: Mapping[str, str],
    parsed_all: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Convert a proof tree to a bipartite fact/apply_rule graph."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    node_ids: dict[tuple[Any, ...], str] = {}
    conclusions = _conclusion_lookup(parsed_all)
    apply_counter = 0
    derived_counter = 0

    def visit(current: Proof, forced_conclusion: Optional[str] = None) -> str:
        nonlocal apply_counter, derived_counter
        signature = proof_signature(current)
        if current.kind == "fact":
            node_id = f"fact:{current.identifier}"
            if signature not in node_ids:
                nodes.append(
                    {
                        "id": node_id,
                        "type": "const",
                        "fact_id": current.identifier,
                        "text": facts.get(current.identifier, current.identifier),
                    }
                )
                node_ids[signature] = node_id
            return node_id

        if signature in node_ids:
            return node_ids[signature]
        premise_ids = [visit(premise) for premise in current.premises]
        apply_counter += 1
        op_id = f"apply:{apply_counter}"
        nodes.append(
            {
                "id": op_id,
                "type": "apply_rule",
                "rule_id": current.identifier,
                "text": rules.get(current.identifier, current.identifier),
            }
        )
        for order, premise_id in enumerate(premise_ids):
            edges.append(
                {
                    "source": premise_id,
                    "target": op_id,
                    "relation": "premise",
                    "order": order,
                }
            )
        derived_counter += 1
        result_id = f"derived:{derived_counter}"
        result_text = forced_conclusion or conclusions.get(signature) or result_id
        nodes.append({"id": result_id, "type": "const", "text": result_text})
        edges.append({"source": op_id, "target": result_id, "relation": "result"})
        node_ids[signature] = result_id
        return result_id

    root_id = visit(proof, conclusion)
    # The recursive lookup usually labels the root already; enforce the gold literal.
    for node in nodes:
        if node["id"] == root_id:
            node["text"] = conclusion
            break
    return {"nodes": nodes, "edges": edges, "root": root_id}


def proof_to_reasoning(
    proof: Proof,
    conclusion: str,
    facts: Mapping[str, str],
    rules: Mapping[str, str],
    parsed_all: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """Render rule applications in deterministic, post-order natural language."""

    conclusions = _conclusion_lookup(parsed_all)
    steps: list[str] = []

    def visit(current: Proof, forced_conclusion: Optional[str] = None) -> str:
        if current.kind == "fact":
            return facts.get(current.identifier, current.identifier)
        premise_texts = [visit(premise) for premise in current.premises]
        result = forced_conclusion or conclusions.get(proof_signature(current), "the result")
        rule_text = rules.get(current.identifier, current.identifier)
        steps.append(
            f"Given {'; '.join(premise_texts)}, apply {current.identifier} "
            f"({rule_text}) to conclude {result}"
        )
        return result

    final = visit(proof, conclusion)
    if not steps:
        steps.append(f"{final} is given directly as {proof.identifier}.")
    return steps


def build_master_row(row: Mapping[str, Any], split: Optional[str] = None) -> dict[str, Any]:
    """Transform one filtered Hugging Face row into the complete master schema."""

    facts, rules = index_theory(row["theory"], row.get("NFact"), row.get("NRule"))
    proved_literal, alternatives, parsed_all = select_gold_proofs(
        row["allProofs"], row["question"], row["answer"]
    )
    rendered_proofs = []
    for alternative in alternatives:
        tree = alternative["tree"]
        rendered_proofs.append(
            {
                "expression": alternative["expression"],
                "tree": tree.to_dict(),
                "depth": alternative["depth"],
                "reasoning": proof_to_reasoning(
                    tree, proved_literal, facts, rules, parsed_all
                ),
                "graph": proof_to_graph(
                    tree, proved_literal, facts, rules, parsed_all
                ),
                "_tree": tree,
            }
        )
    selected = canonical_proof(rendered_proofs)
    source_id = str(row.get("id", ""))
    digest = hashlib.sha1(
        f"{source_id}\0{row['question']}".encode("utf-8")
    ).hexdigest()[:12]
    result = {
        "id": f"{source_id}:{digest}",
        "source_id": source_id,
        "split": split,
        "config": row.get("config"),
        "theory": row["theory"],
        "facts": facts,
        "rules": rules,
        "question": row["question"],
        "answer": row["answer"],
        "depth": int(row["QDep"]),
        "proved_literal": proved_literal,
        "gold_proofs": [
            {key: value for key, value in proof.items() if key != "_tree"}
            for proof in rendered_proofs
        ],
        "canonical_proof": selected["expression"],
        "canonical_proof_tree": selected["tree"],
        "canonical_reasoning": selected["reasoning"],
        "canonical_graph": selected["graph"],
    }
    return result


def filter_rows(
    rows: Iterable[Mapping[str, Any]],
    proofwriter_config: str = "depth-5",
    answers: Sequence[str] = ("True", "False"),
    depths: Sequence[int] = tuple(range(6)),
) -> Iterator[Mapping[str, Any]]:
    allowed_answers = {answer.casefold() for answer in answers}
    allowed_depths = set(depths)
    for row in rows:
        if (
            str(row.get("config", "")).casefold() == proofwriter_config.casefold()
            and str(row.get("answer", "")).casefold() in allowed_answers
            and int(row.get("QDep", -1)) in allowed_depths
        ):
            yield row


def _balanced_quotas(
    size: int,
    answers: Sequence[str],
    depths: Sequence[int],
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], int]]:
    if size < 0:
        raise ValueError("Sample size must be non-negative")
    answer_keys = [answer.casefold() for answer in answers]
    strata = [(answer, depth) for answer in answer_keys for depth in depths]
    if not strata:
        if size:
            raise ValueError("Cannot sample from an empty set of strata")
        return [], {}

    base, remainder = divmod(size, len(strata))
    quotas = {key: base for key in strata}
    answer_extras = {answer: 0 for answer in answer_keys}
    depth_extras = {depth: 0 for depth in depths}
    available = list(strata)
    for _ in range(remainder):
        key = min(
            available,
            key=lambda item: (
                answer_extras[item[0]],
                depth_extras[item[1]],
                strata.index(item),
            ),
        )
        quotas[key] += 1
        answer_extras[key[0]] += 1
        depth_extras[key[1]] += 1
        available.remove(key)
    return strata, quotas


def balanced_sample(
    rows: Iterable[Mapping[str, Any]],
    size: int,
    seed: int = 42,
    answers: Sequence[str] = ("True", "False"),
    depths: Sequence[int] = tuple(range(6)),
) -> list[Mapping[str, Any]]:
    """Deterministically sample approximately equally across answer × QDep."""

    strata, quotas = _balanced_quotas(size, answers, depths)
    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["answer"]).casefold(), int(row["QDep"]))
        if key in strata:
            buckets[key].append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen: list[Mapping[str, Any]] = []
    offsets: dict[tuple[str, int], int] = {}
    for key in strata:
        take = min(quotas[key], len(buckets[key]))
        chosen.extend(buckets[key][:take])
        offsets[key] = take

    while len(chosen) < size:
        candidates = [
            key for key in strata if offsets[key] < len(buckets[key])
        ]
        if not candidates:
            break
        key = min(candidates, key=lambda item: (offsets[item], strata.index(item)))
        chosen.append(buckets[key][offsets[key]])
        offsets[key] += 1

    rng.shuffle(chosen)
    return chosen


def balanced_reservoir_sample(
    rows: Iterable[Mapping[str, Any]],
    size: int,
    seed: int = 42,
    answers: Sequence[str] = ("True", "False"),
    depths: Sequence[int] = tuple(range(6)),
) -> list[Mapping[str, Any]]:
    """Stratified reservoir sample without materializing the input iterable.

    Every answer × depth stratum receives either ``size // n_strata`` or one
    additional row.  This is intended for Hugging Face ``IterableDataset``
    splits, where collecting every depth-5 row would be unnecessarily costly.
    """

    strata, quotas = _balanced_quotas(size, answers, depths)
    reservoirs: dict[tuple[str, int], list[Mapping[str, Any]]] = {
        key: [] for key in strata
    }
    seen = {key: 0 for key in strata}
    rng = random.Random(seed)

    for row in rows:
        key = (str(row.get("answer", "")).casefold(), int(row.get("QDep", -1)))
        if key not in quotas or quotas[key] == 0:
            continue
        seen[key] += 1
        bucket = reservoirs[key]
        if len(bucket) < quotas[key]:
            bucket.append(row)
        else:
            replacement = rng.randrange(seen[key])
            if replacement < quotas[key]:
                bucket[replacement] = row

    shortages = {
        f"{answer}/{depth}": quotas[(answer, depth)] - len(reservoirs[(answer, depth)])
        for answer, depth in strata
        if len(reservoirs[(answer, depth)]) < quotas[(answer, depth)]
    }
    if shortages:
        raise ValueError(f"Not enough rows for balanced sample: {shortages}")

    chosen = [row for key in strata for row in reservoirs[key]]
    rng.shuffle(chosen)
    return chosen


def load_official_splits(
    dataset_name: str = "tasksource/proofwriter",
    dataset_config: str = "default",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> dict[str, Iterable[Mapping[str, Any]]]:
    """Load only ProofWriter's official train and test splits via datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to download ProofWriter"
        ) from exc
    loaded = load_dataset(
        dataset_name,
        name=dataset_config,
        split=["train", "test"],
        cache_dir=cache_dir,
        streaming=streaming,
    )
    return {"train": loaded[0], "test": loaded[1]}


def write_json(path: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write master rows as a UTF-8 JSON array."""

    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
