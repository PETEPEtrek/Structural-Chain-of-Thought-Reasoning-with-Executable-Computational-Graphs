# Structural Chain-of-Thought Reasoning with Executable Computational Graphs

Instead of emitting free-form chain-of-thought, the model emits an **executable
computational graph**: a set of named nodes, each with an operation and
arguments, whose final node is evaluated by an interpreter to produce the
answer. Because the answer is *computed* rather than *asserted*, the reasoning
is checkable — we can execute the graph, and we can intervene on it and check
that the model's answer moves accordingly (faithfulness).

```
datasets/    all six datasets, one file each, {"train": [...], "test": [...]}
data_prep/   how those files were built from the raw sources
training/    SFT, default GRPO, graph GRPO
inference/   inference + faithfulness evaluation (each script self-contained)
prompts/     few-shot exemplar prompts used by the ICL runs
misc/        plotting
```

---

## Table of Contents

1. [Extended Version of the Paper](#extended-version-of-the-paper)
2. [Environment Setup](#environment-setup)
3. [Datasets](#datasets)
4. [Training](#training)
5. [Inference](#inference)
6. [Faithfulness](#faithfulness)
7. [Data preparation](#data-preparation)
8. [Misc](#misc)
9. [Citation](#citation)
10. [Contact](#contact)

---

## Extended Version of the Paper

The extended version of the paper, including appendices, is available on **arXiv**:
[Structural Chain-of-Thought: Reasoning with Executable Computational Graphs](https://arxiv.org/abs/2610.15107)

## Environment Setup

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended; inference uses vLLM, training uses PyTorch)

### Installation

Create a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n scotrg python=3.12
conda activate scotrg
```

Install dependencies. `torch` in `requirements.txt` is the CUDA 12.4 build from conda env `peter2`, so install it from the PyTorch wheel index first:

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Or using uv (faster, if available)
uv pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

BLEURT is optional and only needed for the official EntailmentBank intermediate scorer (`--bleurt-checkpoint`). The default evaluator is lexical and does not require it.

## Datasets

Every file is a single JSON object with `train` and `test` keys.

| File | Train | Test | Task |
|---|---|---|---|
| `datasets/gsm8k.json` | 5656 | 1415 | GSM8K grade-school word problems, with graph annotations |
| `datasets/gsm8k_new_ops.json` | 5634 | 1409 | GSM8K+ — same problems, arithmetic operators **redefined** |
| `datasets/igsm.json` | 2634 | 658 | iGSM synthetic multi-step arithmetic |
| `datasets/math.json` | 7959 | 1992 | MATH competition problems |
| `datasets/entailment_bank.json` | 1313 | 340 | EntailmentBank task 2, proofs as entailment graphs |
| `datasets/proofwriter.json` | 5000 | 2000 | ProofWriter logical deduction with proofs |

The graph is a JSON array of nodes, each either a constant or an operation,
listed in computational order; the value of the last node is the answer:

```json
[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]
```

Every model — ICL, graph SFT
and graph GRPO — is prompted for and evaluated on exactly this form, so all
numbers are directly comparable.

## Training

Each `training/<method>/` directory is self-sufficient: the entry-point script
plus the prompt and helper modules it imports. Run from inside the directory.

| Method | Math | Entailment | ProofWriter |
|---|---|---|---|
| **SFT** | `graph_sft_script.py` | `entailment_graph_sft_script.py`, `entailment_baseline_sft_script.py` | `proofwriter_sft_script.py` |
| **Default GRPO** (style and answer-only reward) | `default_grpo_script.py` | `entailment_default_grpo_script.py` | `proofwriter_default_grpo_script.py` |
| **Graph GRPO** (graph-structure reward) | `grpo_computational_script.py` | `entailment_graph_grpo_script.py` | `proofwriter_graph_grpo_script.py` |

```bash
cd training/graph_grpo
python grpo_computational_script.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-file ../../datasets/gsm8k_new_ops.json \
  --output-dir ./ckpt --save-path ./ckpt/final
```

## Inference

Every script in `inference/` is **standalone** — only pip packages
(`vllm`, `transformers`, `torch`, `peft`, `tqdm`), no imports from sibling files.
Copy one out on its own and it still runs. All of them accept
`--test-file` together with `--split {train,test}` to pick a split out of the
bundled dataset files.

| Script | What it evaluates |
|---|---|
| `infer_sft.py` | a graph-SFT checkpoint |
| `infer_default_grpo.py` | a default-GRPO checkpoint (free-form CoT) |
| `infer_graph_grpo.py` | a graph-GRPO checkpoint |
| `infer_icl.py` | base model, few-shot prompted (no training) |
| `infer_tot.py` | Tree-of-Thought beam search baseline |
| `infer_entailment.py` | EntailmentBank, with the official scorer |
| `infer_proofwriter.py` | ProofWriter: answer accuracy, proof exact match, leaf/step F1 |

```bash
python inference/infer_graph_grpo.py \
  --model-base-path Qwen/Qwen3-4B-Instruct-2507 \
  --checkpoint-path /path/to/lora \
  --test-file datasets/igsm.json --split test \
  --results-dir results --results-suffix qwen_graph_grpo_igsm
```

Pass an empty `--checkpoint-path ""` to evaluate the base model without a LoRA
adapter. `infer_icl.py` and `infer_tot.py` take no checkpoint and instead
require the dataset identity, because prompt and answer handling differ per
dataset (`--dataset-name` / `--dataset`, one of `gsm8k`, `gsm8k_new_ops`,
`igsm`, `math`; `infer_tot.py` also accepts `entailment`):

```bash
python inference/infer_icl.py \
  --model-base-path Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-name gsm8k_new_ops \
  --test-file datasets/gsm8k_new_ops.json --split test \
  --prompts-json prompts/graph_inference_prompts.json \
  --results-dir results --results-suffix qwen_icl_gsm8k_new_ops
```

## Faithfulness

Following Somov et al., faithfulness is measured by **intervention**: mutate a
node in the model's own graph, re-ask, and check that the answer follows the
mutated graph.

| Metric | Definition |
|---|---|
| `F_ID` | the emitted answer equals the execution of the model's own graph |
| `F_OOD` | after intervention, the emitted answer equals the execution of the mutated graph |
| `F_strong` | both hold on the same sample |

| Script | Datasets |
|---|---|
| `eval_faithfulness.py` | GSM8K, GSM8K+, iGSM, MATH |
| `eval_faithfulness_entailment.py` | EntailmentBank |
| `eval_faithfulness_proofwriter.py` | ProofWriter |

Two modes. `--mode offline` recomputes the metrics from an existing inference
JSON, with no GPU:

```bash
python inference/eval_faithfulness.py --mode offline \
  --inference-json results/qwen_graph_grpo_igsm.json
```

`--mode online` runs generation itself, including the intervention pass:

```bash
python inference/eval_faithfulness.py --mode online \
  --model-base-path Qwen/Qwen3-4B-Instruct-2507 \
  --checkpoint-path "" \
  --test-file datasets/gsm8k_new_ops.json --split test \
  --dataset-name gsm8k_new_ops \
  --prompts-json prompts/graph_inference_prompts.json --has-answer \
  --output-dir results --output-stem qwen_icl_gsm8k_new_ops
```

## Data preparation

`data_prep/` reconstructs `datasets/` from the raw sources. Graph annotations
for iGSM and MATH were generated with an LLM through OpenRouter, so those two
scripts need `OPENROUTER_API_KEY` and their output is not bit-reproducible.

| Script | Produces |
|---|---|
| `prepare_gsm8k_new_ops_splits.py` | `gsm8k_new_ops.json` (applies the redefined operators and recomputes gold answers) |
| `generate_graphs_openrouter_igsm.py` | `igsm.json` |
| `generate_graphs_openrouter_math.py` | `math.json` |
| `convert_entailmentbank_to_graphs.py` | `entailment_bank.json` |
| `prepare_entailmentbank_baseline_data.py` | plain-text entailment data for the baseline |
| `prepare_proofwriter_data.py` | `proofwriter.json` |
| `prepare_proofwriter_training_data.py` | ProofWriter train files for SFT/GRPO |

## Misc

`misc/tokens_vs_acc_plot.py` — accuracy against generated-token budget.

---

## Citation

If you use this work in your research, please cite the following paper:

```bibtex
@misc{senatorov2026structuralchainofthoughtreasoningexecutable,
      title={Structural Chain-of-Thought: Reasoning with Executable Computational Graphs},
      author={Petr Senatorov and Karim Vafin and Elena Tutubalina and Oleg Somov},
      year={2026},
      eprint={2610.15107},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.15107},
}
```

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Telegram:** [@PETEPEtrek](https://t.me/PETEPEtrek)
- **Email:** [senatorov.pp@phystech.edu](mailto:senatorov.pp@phystech.edu)
