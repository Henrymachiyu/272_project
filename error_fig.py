#!/usr/bin/env python3
"""Error breakdown on test (5,500 queries) under the shared team protocol, as a slide image.

    python common_eval.py && python error_fig.py   # writes results/common/fig_llm_errors.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CATS = [("wrong intent", "wrong_intent", "#2a78d6"), ("in-scope refused", "refused_in", "#eb6834"),
        ("OOS accepted", "accepted_oos", "#1baf7a")]
R = json.load(open(os.path.join(HERE, "results", "common", "common_report.json")))
ROWS = [("Zero-shot, 151", "zs_151"), ("LoRA, 151", "lora_151"),
        ("Zero-shot, 150 + τ", "zs_150"), ("LoRA, 150 + τ", "lora_150")]


def bd(k):
    return R[k]["breakdown"] if k.startswith("zs") else R[k]["breakdown_mean"]


plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(figsize=(9.5, 3.8))
names = ROWS[::-1]
xmax = max(sum(bd(k)[c] for _, c, _ in CATS) for _, k in ROWS)
for y, (n, k) in enumerate(names):
    left = 0
    for i, (c, key, col) in enumerate(CATS):
        v = bd(k)[key]
        ax.barh(y, v, left=left, color=col, height=0.6, edgecolor=SURF, linewidth=2,
                label=c if y == len(names) - 1 else None)
        if v > 0.07 * xmax:
            ax.text(left + v / 2, y, f"{v:,.0f}", ha="center", va="center", color="white", fontsize=11.5,
                    fontweight="bold")
        left += v
    ax.text(left + 0.012 * xmax, y, f"{left:,.0f}", va="center", color=INK, fontsize=12.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n for n, _ in names], fontsize=13, color=INK)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)
ax.set_xlim(0, xmax * 1.12)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("errors on 5,500 test queries (LoRA: mean over 3 seeds)", color=INK2)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.18), ncol=3, frameon=False)
fig.tight_layout()
out = os.path.join(HERE, "results", "common", "fig_llm_errors.png")
fig.savefig(out, dpi=220)
print(out)
