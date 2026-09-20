#!/usr/bin/env python3
"""
Two-panel results figure: accuracy (top) and token cost (bottom, log scale),
grouped by dataset, one bar per training method.

Reads two JSON files from the same directory as this script:
    performance.json   {dataset: {method: {model: {"value": x, "std": s}}}}
    tokens.json        same shape, values in tokens

Writes results_figure.pdf / .png next to them.

    python plot_results.py                       # single column (ACL, 3.15in)
    python plot_results.py --width full          # two-column span
    python plot_results.py --err seeds           # error bars = seed std
    python plot_results.py --all-methods         # keep ToT and Answer-only SFT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Methods drawn in the body figure, in bar order. Any method present in the
# JSON but absent here is skipped unless --all-methods is given.
BODY_METHODS = ["ICL with FS", "CoT + Answer SFT", "Graph SFT",
                "Default GRPO", "Graph-Based GRPO"]
EXTRA_METHODS = ["Tree-of-Thoughts", "Answer-only SFT"]

# The MATH block names Graph SFT differently; fold the variants together.
ALIASES = {"Graph W Answer SFT": "Graph SFT",
           "Default GRO": "Default GRPO"}

SHORT = {"ICL with FS": "ICL", "Tree-of-Thoughts": "ToT",
         "Answer-only SFT": "Answer SFT", "CoT + Answer SFT": "CoT+Ans SFT",
         "Graph SFT": "Graph SFT", "Default GRPO": "Def. GRPO",
         "Graph-Based GRPO": "Graph GRPO"}

COLOR = {"ICL with FS": "#9e9e9e", "Tree-of-Thoughts": "#c49a6c",
         "Answer-only SFT": "#bdbdbd", "CoT + Answer SFT": "#e8a33d",
         "Graph SFT": "#4f9ad7", "Default GRPO": "#8c6bb1",
         "Graph-Based GRPO": "#d1495b"}


def load(path):
    """JSON -> {dataset: {method: {"vals": [...], "stds": [...]}}}"""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for ds, methods in raw.items():
        out[ds] = {}
        for method, models in methods.items():
            name = ALIASES.get(method, method)
            if name in out[ds]:
                print(f"warning: {ds}/{method} collides with an existing "
                      f"'{name}' entry; keeping the first", file=sys.stderr)
                continue
            names = sorted(models)
            out[ds][name] = {
                "models": names,
                "vals": np.array([models[m]["value"] for m in names], float),
                "stds": np.array([models[m].get("std", 0.0) for m in names],
                                 float),
            }
    return out


def agg(entry, mode):
    """Bar height and error bar for one dataset x method cell."""
    mean = entry["vals"].mean()
    if mode == "models":          # spread across models
        err = entry["vals"].std(ddof=0)
    elif mode == "seeds":         # mean seed-level std
        err = entry["stds"].mean()
    else:                         # both, added in quadrature
        err = np.sqrt(entry["vals"].std(ddof=0) ** 2
                      + (entry["stds"] ** 2).mean())
    return mean, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perf", default=HERE / "performance_dict.json")
    p.add_argument("--tokens", default=HERE / "tokens_consumption.json")
    p.add_argument("--out", default=HERE / "results_figure")
    p.add_argument("--width", choices=["col", "full"], default="col")
    p.add_argument("--err", choices=["models", "seeds", "both"],
                   default="models")
    p.add_argument("--all-methods", action="store_true")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="explicit dataset order; default = order in the JSON")
    args = p.parse_args()

    acc, tok = load(args.perf), load(args.tokens)

    datasets = args.datasets or list(acc)
    missing = [d for d in datasets if d not in tok]
    if missing:
        sys.exit(f"datasets missing from tokens file: {missing}")

    methods = BODY_METHODS + (EXTRA_METHODS if args.all_methods else [])
    methods = [m for m in methods
               if any(m in acc[d] and m in tok[d] for d in datasets)]
    if not methods:
        sys.exit("no methods found in both files")

    figw = 3.15 if args.width == "col" else 6.6
    fs = 6.5 if args.width == "col" else 9
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(figw, 2.9 if args.width == "col" else 4.0),
        sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))

    n = len(methods)
    w = 0.8 / n
    xs = np.arange(len(datasets))

    for i, m in enumerate(methods):
        off = (i - (n - 1) / 2) * w
        av, ae, tv = [], [], []
        for d in datasets:
            if m in acc[d]:
                mu, er = agg(acc[d][m], args.err)
            else:
                mu, er = np.nan, np.nan          # gap instead of a fake bar
            av.append(mu)
            ae.append(er)
            tv.append(agg(tok[d][m], args.err)[0] if m in tok[d] else np.nan)
        a1.bar(xs + off, av, w, yerr=ae, color=COLOR.get(m),
               label=SHORT.get(m, m),
               error_kw=dict(elinewidth=0.6, capsize=0, ecolor="#555"))
        a2.bar(xs + off, tv, w, color=COLOR.get(m))

    a1.set_ylabel("Accuracy", fontsize=fs + 0.5)
    a2.set_ylabel("Tokens\n(lower better)", fontsize=fs + 0.5)
    a2.set_yscale("log")

    finite = [v for ax in (a2,) for c in ax.containers for b in c
              for v in [b.get_height()] if np.isfinite(v) and v > 0]
    if finite:
        a2.set_ylim(max(1, min(finite) * 0.55), max(finite) * 1.8)

    a2.set_xticks(xs)
    a2.set_xticklabels(datasets, fontsize=fs, rotation=18, ha="right")
    for ax in (a1, a2):
        ax.tick_params(axis="y", labelsize=fs)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ncol = 3 if n <= 6 else 4
    a1.legend(fontsize=fs - 0.9, frameon=False, ncol=ncol, loc="upper center",
              bbox_to_anchor=(0.5, 1.30 + 0.12 * (n > 6)),
              columnspacing=1.0, handlelength=1.1)

    fig.tight_layout(h_pad=0.4)
    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    print(f"datasets: {datasets}")
    print(f"methods:  {methods}")
    print(f"error bars: {args.err}")


if __name__ == "__main__":
    main()
