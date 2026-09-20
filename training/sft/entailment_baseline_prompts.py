"""
Prompts and input builders for EntailmentBank baseline training
(SFT answer, SFT CoT+answer, default GRPO).

Gold proof format (native EntailmentBank):
  sent1 & sent3 -> int1: leo is a constellation containing stars; int1 & sent2 -> hypothesis;
"""

import re

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

PROOF_STEP_RE = re.compile(r"\s*;\s*")

SYSTEM_PROMPT_DEFAULT_GRPO = (
    "You are given a science question and supporting context facts.\n"
    "Think about the question and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    "Then, provide the entailment proof between "
    f"{SOLUTION_START}{SOLUTION_END}.\n"
    "Proof format (one or more steps separated by ';'):\n"
    "sentA & sentB -> int1: intermediate conclusion; int1 & sentC -> hypothesis;"
)

SYSTEM_PROMPT_ANSWER_SFT = (
    "Given a science question and context facts, produce an entailment proof.\n"
    "Proof format (steps separated by ';'):\n"
    "sentA & sentB -> int1: intermediate conclusion; int1 & sentC -> hypothesis;"
)

SYSTEM_PROMPT_COT_ANSWER_SFT = (
    "Given a science question and context facts, first explain the reasoning, "
    "then provide the entailment proof after ANSWER:.\n"
    "Proof format:\n"
    "sentA & sentB -> int1: intermediate conclusion; int1 & sentC -> hypothesis;"
)


def build_user_input(item: dict) -> str:
    parts = [f"question: {item.get('question', '').strip()}"]
    context = (item.get("context") or "").strip()
    if context:
        parts.append(f"context: {context}")
    return "\n".join(parts)


def build_proof_target(item: dict) -> str:
    return (item.get("proof") or "").strip()


def build_answer_target(item: dict) -> str:
    """SFT answer mode: compact native proof string."""
    return build_proof_target(item)


def build_cot_plus_answer_target(item: dict) -> str:
    cot = (item.get("full_text_proof") or "").strip()
    proof = build_proof_target(item)
    if cot and proof:
        return f"{cot}\nANSWER: {proof}"
    if proof:
        return proof
    return cot


def normalize_proof(text: str) -> str:
    value = (text or "").strip().lower()
    value = PROOF_STEP_RE.sub("; ", value)
    value = re.sub(r"\s*->\s*", " -> ", value)
    value = re.sub(r"\s*&\s*", " & ", value)
    value = re.sub(r"\s*:\s*", ": ", value)
    while value.endswith(";"):
        value = value[:-1].strip()
    return " ".join(value.split())


# Backward-compatible alias used by GRPO reward code.
normalize_answer = normalize_proof


def get_system_prompt(method: str) -> str:
    mapping = {
        "answer_sft": SYSTEM_PROMPT_ANSWER_SFT,
        "cot_plus_answer_sft": SYSTEM_PROMPT_COT_ANSWER_SFT,
        "default_grpo": SYSTEM_PROMPT_DEFAULT_GRPO,
        "icl_answer": SYSTEM_PROMPT_ANSWER_SFT,
        "icl_cot_plus_answer": SYSTEM_PROMPT_COT_ANSWER_SFT,
        "icl_default_grpo": SYSTEM_PROMPT_DEFAULT_GRPO,
    }
    if method not in mapping:
        raise ValueError(f"Unknown method: {method}")
    return mapping[method]


def build_assistant_target(item: dict, method: str) -> str:
    if method in {"answer_sft", "icl_answer"}:
        return build_answer_target(item)
    if method in {"cot_plus_answer_sft", "icl_cot_plus_answer"}:
        return build_cot_plus_answer_target(item)
    if method in {"default_grpo", "icl_default_grpo"}:
        proof = build_proof_target(item)
        cot = (item.get("full_text_proof") or "").strip()
        if cot:
            return (
                f"{REASONING_START}\n{cot}\n{REASONING_END}\n"
                f"{SOLUTION_START}{proof}{SOLUTION_END}"
            )
        return f"{SOLUTION_START}{proof}{SOLUTION_END}"
    raise ValueError(f"Unknown method: {method}")


def extract_proof_from_response(text: str, method: str) -> str:
    response = (text or "").strip()
    if not response:
        return ""

    if method in {"cot_plus_answer_sft", "icl_cot_plus_answer"}:
        upper = response.upper()
        marker = "ANSWER:"
        idx = upper.rfind(marker)
        if idx >= 0:
            return response[idx + len(marker) :].strip()

    if method in {"default_grpo", "icl_default_grpo"}:
        match = re.search(
            re.escape(SOLUTION_START) + r"(.*?)" + re.escape(SOLUTION_END),
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

    # answer_sft / icl_answer / fallback: pick longest proof-like span
    if "->" in response:
        lines = []
        for raw in response.splitlines():
            line = raw.strip()
            if not line or line.startswith("```"):
                continue
            if "->" in line:
                lines.append(line.rstrip(";"))
        if lines:
            return "; ".join(lines) + (";" if lines else "")
    return response.strip()


def is_icl_method(method: str) -> bool:
    return method.startswith("icl_")


MODEL_BASE_MAP = {
    "llama3_2": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "falcon3": "tiiuae/Falcon3-3B-Instruct",
    "gemma_2_2b": "google/gemma-2-2b-it",
}


def _adapter_slug(path_like: str) -> str:
    return path_like.lower().replace("\\", "/").strip("/")


def infer_method_from_adapter_name(name: str) -> str:
    slug = _adapter_slug(name)
    if "cot_plus_answer" in slug:
        return "cot_plus_answer_sft"
    if "default_grpo" in slug:
        return "default_grpo"
    if "_answer_lora" in slug or "_answer_" in slug:
        return "answer_sft"
    raise ValueError(f"Cannot infer method from adapter name: {name}")


def infer_model_alias_from_adapter_name(name: str) -> str:
    slug = _adapter_slug(name)
    for alias in sorted(MODEL_BASE_MAP.keys(), key=len, reverse=True):
        if slug.startswith(alias):
            return alias
    raise ValueError(f"Cannot infer model alias from adapter name: {name}")


def infer_seed_from_adapter_name(name: str):
    import re
    m = re.search(r"seed(\d+)", _adapter_slug(name))
    return int(m.group(1)) if m else None


def is_adapter_dir(path) -> bool:
    from pathlib import Path
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file()


def discover_adapter_dirs(adapters_root, seed_filter=None, model_filter=None):
    from pathlib import Path
    root = Path(adapters_root)
    if not root.is_dir():
        return []

    found = []
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    seen = set()
    for cand in candidates:
        if not is_adapter_dir(cand):
            continue
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        rel = cand.relative_to(root) if cand != root else cand
        slug = _adapter_slug(str(rel).replace("/", "_"))
        if seed_filter is not None:
            s = infer_seed_from_adapter_name(slug)
            if s is not None and s != seed_filter:
                continue
        if model_filter:
            alias = infer_model_alias_from_adapter_name(slug)
            if alias != model_filter:
                continue
        method = infer_method_from_adapter_name(slug)
        alias = infer_model_alias_from_adapter_name(slug)
        found.append(
            {
                "path": str(cand),
                "slug": slug,
                "method": method,
                "model_alias": alias,
                "model_base": MODEL_BASE_MAP[alias],
                "seed": infer_seed_from_adapter_name(slug),
            }
        )

    found.sort(key=lambda x: (x["model_alias"], x["method"], x["path"]))
    return found
