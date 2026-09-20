#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

import graph_prompts


MAX_SEQ_LENGTH = 1024
LORA_RANK = 32

GEMMA_CHAT_TEMPLATE_WITH_SYSTEM = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"""


DATASET_FILE_MAP = {
    "gsm8k": {
        "sft": "gsm8k_comp_graphs_train.json",
        "pre_grpo": "gsm8k_comp_graphs_sft_train.json",
    },
    "igsm": {
        "sft": "igsm_openrouter_valid_train.json",
        "pre_grpo": "igsm_openrouter_valid_train_sft.json",
    },
    "math": {
        "sft": "math_openrouter_valid_train.json",
        "pre_grpo": "math_openrouter_valid_train_sft.json",
    },
    "gsm8k_new_ops": {
        "sft": "gsm8k_new_ops_comp_graphs_train.json",
        "pre_grpo": "gsm8k_new_ops_comp_graphs_sft_train.json",
    },
}


def extract_hash_answer(text: str):
    return text.split("####", 1)[1].strip() if "####" in text else None


def resolve_model_alias(model_path: str) -> str:
    path = model_path.lower()
    if "qwen2.5-3b" in path or "qwen2_5" in path:
        return "qwen2_5_3b"
    if "qwen3-4b" in path or "qwen3_4b" in path:
        return "qwen3_4b"
    if "qwen3" in path:
        return "qwen3"
    if "gemma-2-2b" in path or "gemma_2_2b" in path:
        return "gemma_2_2b"
    if "llama" in path:
        return "llama3_2"
    if "falcon" in path:
        return "falcon3"
    return "model"


def is_gemma_2_2b(path_like: str) -> bool:
    path = path_like.lower()
    return "gemma-2-2b" in path or "gemma_2_2b" in path


def maybe_set_gemma_chat_template(tokenizer, model_path: str) -> None:
    if is_gemma_2_2b(model_path):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE_WITH_SYSTEM


def resolve_dataset_path(
    dataset_name: str,
    stage: str,
    dataset_path: str | None,
) -> str:
    if dataset_path:
        path = dataset_path
    else:
        path = DATASET_FILE_MAP[dataset_name][stage]
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"(dataset={dataset_name}, stage={stage}). "
            "Pass --dataset-path or generate splits with the data_prep/ scripts."
        )
    return str(resolved)


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_graph_text(example: dict) -> str:
    return json.dumps(example["computational_graph"], indent=2)


def get_question_text(example: dict, dataset_name: str) -> str:
    if dataset_name == "igsm":
        # OpenRouter IGSM files use "question"; keep backward compatibility.
        return example.get("question") or example.get("problem", "")
    return example.get("question", "")


def get_target_answer(example: dict, dataset_name: str) -> str | None:
    if dataset_name in {"igsm", "math", "gsm8k_new_ops"}:
        answer = example.get("answer")
        return None if answer is None else str(answer).strip()
    return extract_hash_answer(example.get("solution", ""))


def select_graph_system_prompt(use_cot: bool, has_answer: bool) -> str:
    if use_cot and has_answer:
        return graph_prompts.SYSTEM_PROMPT_FULL_W_ANSWER
    if use_cot and not has_answer:
        return graph_prompts.SYSTEM_PROMPT_FULL_WO_ANSWER
    if has_answer:
        return graph_prompts.SYSTEM_PROMPT_W_ANSWER
    return graph_prompts.SYSTEM_PROMPT


def build_graph_completion(
    example: dict,
    graph_text: str,
    use_cot: bool,
    has_answer: bool,
    dataset_name: str,
) -> str:
    if use_cot:
        text = graph_prompts.build_solution_with_graph(example["solution"], graph_text)
        if has_answer:
            answer = get_target_answer(example, dataset_name)
            return f"{text}\nANSWER: {answer}"
        return text
    if has_answer:
        answer = get_target_answer(example, dataset_name)
        return f"{graph_text}\nANSWER: {answer}"
    return graph_text


def preprocess_dataset(
    mode: str,
    data: list[dict],
    dataset_name: str,
    use_cot: bool,
    has_answer: bool,
) -> Dataset:
    processed = []

    if mode == "answer_sft":
        for item in data:
            answer = get_target_answer(item, dataset_name)
            question_text = get_question_text(item, dataset_name)
            if not question_text or answer is None:
                continue
            processed.append(
                {
                    "prompt": [
                        {"role": "user", "content": question_text}
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    ],
                }
            )
        if not processed:
            raise ValueError(
                "SFT dataset is empty after preprocessing. "
                f"mode={mode}, dataset_name={dataset_name}"
            )
        return Dataset.from_list(processed)

    if mode == "cot_plus_answer_sft":
        for item in data:
            question_text = get_question_text(item, dataset_name)
            if not question_text:
                continue
            processed.append(
                {
                    "prompt": [
                        {"role": "user", "content": question_text}
                    ],
                    "completion": [
                        {"role": "assistant", "content": item["solution"]}
                    ],
                }
            )
        if not processed:
            raise ValueError(
                "SFT dataset is empty after preprocessing. "
                f"mode={mode}, dataset_name={dataset_name}"
            )
        return Dataset.from_list(processed)

    system_prompt = select_graph_system_prompt(use_cot=use_cot, has_answer=has_answer)
    for item in data:
        question_text = get_question_text(item, dataset_name)
        answer = get_target_answer(item, dataset_name)
        if not question_text or (has_answer and answer is None):
            continue
        graph_text = get_graph_text(item)
        completion = build_graph_completion(
            item,
            graph_text=graph_text,
            use_cot=use_cot,
            has_answer=has_answer,
            dataset_name=dataset_name,
        )
        processed.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_text},
                ],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    if not processed:
        raise ValueError(
            "SFT dataset is empty after preprocessing. "
            f"mode={mode}, dataset_name={dataset_name}"
        )
    return Dataset.from_list(processed)


def compute_num_epochs(tuning: str, stage: str) -> int:
    if tuning == "lora" and stage == "sft":
        return 3
    if tuning == "lora" and stage == "pre_grpo":
        return 1
    if tuning == "full_tune" and stage == "sft":
        return 10
    if tuning == "full_tune" and stage == "pre_grpo":
        return 3
    raise ValueError(f"Unsupported tuning/stage combination: {tuning}/{stage}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def build_paths(
    model_alias: str,
    mode: str,
    has_answer: bool,
    use_cot: bool,
    tuning: str,
    stage: str,
    dataset_name: str,
    seed: int,
    output_dir: str | None,
    logging_dir: str | None,
    save_model_dir: str | None,
) -> tuple[str, str, str]:
    mode_tag_map = {
        "answer_sft": "answer",
        "cot_plus_answer_sft": "cot_plus_answer",
        "graph_sft": "graphs",
    }
    mode_tag = mode_tag_map[mode]
    answer_tag = "has_answer" if has_answer else "has_no_answer"
    full_tag = "full" if use_cot else "not_full"
    base = (
        f"{model_alias}_sft_{mode_tag}_{answer_tag}/"
        f"{full_tag}_{tuning}/{stage}_{dataset_name}_seed{seed}"
    )
    final_output_dir = output_dir or base
    final_save_model_dir = save_model_dir or base
    final_logging_dir = logging_dir or f"runs_{base}"
    return final_output_dir, final_logging_dir, final_save_model_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurable graph SFT script for GSM8K, GSM8K+, iGSM and MATH."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF model path, e.g. Qwen/Llama/Falcon checkpoint.",
    )
    parser.add_argument(
        "--config",
        choices=["answer_sft", "cot_plus_answer_sft", "graph_sft"],
        required=True,
    )
    parser.add_argument(
        "--has-answer",
        action="store_true",
        help="Used for graph_sft. If enabled, graph completion includes ANSWER.",
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Used for graph_sft. If enabled, use FULL prompts and CoT+graph completion.",
    )
    parser.add_argument(
        "--tuning",
        choices=["lora", "full_tune"],
        default="lora",
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "pre_grpo"],
        default="sft",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "gsm8k",
            "igsm",
            "math",
            "gsm8k_new_ops",
        ],
        default="gsm8k",
        help="Dataset family used for automatic dataset path resolution.",
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--logging-dir")
    parser.add_argument("--save-model-dir")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    dataset_path = resolve_dataset_path(
        dataset_name=args.dataset,
        stage=args.stage,
        dataset_path=args.dataset_path,
    )
    data = load_data(dataset_path)

    if args.config != "graph_sft":
        # Keep path naming consistent even for non-graph configs.
        has_answer = args.config in ["answer_sft", "cot_plus_answer_sft"]
        use_cot = args.config == "cot_plus_answer_sft"
    else:
        has_answer = args.has_answer
        use_cot = args.use_cot

    dataset = preprocess_dataset(
        mode=args.config,
        data=data,
        dataset_name=args.dataset,
        use_cot=use_cot,
        has_answer=has_answer,
    )
    dataset = dataset.shuffle(seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    maybe_set_gemma_chat_template(tokenizer, args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if args.tuning == "lora":
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)

    model_alias = resolve_model_alias(args.model_path)
    output_dir, logging_dir, save_model_dir = build_paths(
        model_alias=model_alias,
        mode=args.config,
        has_answer=has_answer,
        use_cot=use_cot,
        tuning=args.tuning,
        stage=args.stage,
        dataset_name=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_model_dir=args.save_model_dir,
    )

    num_train_epochs = compute_num_epochs(args.tuning, args.stage)
    learning_rate = 3e-4 if args.stage == "pre_grpo" else 5e-6

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=250,
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        bf16=True,
        fp16=False,
        max_grad_norm=0.1,
        save_total_limit=1,
        max_length=args.max_seq_length,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(save_model_dir)

    print("\n✅ Training completed!")
    print(f"Model saved to: {save_model_dir}")


if __name__ == "__main__":
    main()
