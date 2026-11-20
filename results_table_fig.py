#!/usr/bin/env python3
"""Render the zero-shot / LoRA test results (shared team protocol) as a slide image.

    python common_eval.py && python results_table_fig.py   # writes results/common/fig_llm_results_table.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID, SURF, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#eef4fc"

R = json.load(open(os.path.join(HERE, "results", "common", "common_report.json")))
SYSTEMS = [("Zero-shot", "zs_151", "151 labels"), ("LoRA", "lora_151", "151 labels"),
           ("Zero-shot", "zs_150", "150 labels + threshold"), ("LoRA", "lora_150", "150 labels + threshold")]
COLS = [("in-scope\nacc", "in_acc"), ("OOS\nrecall", "oos_recall"), ("OOS\nprecision", "oos_precision"),
        ("OOS\nF1", "oos_f1"), ("in-scope\nmacro-F1", "in_macro_f1"), ("AUROC", "auroc")]


def cell(key, m):
    if key.startswith("zs"):
        return 100 * R[key]["test"][m], None
    mean, sd = R[key]["mean_std"][m]
    return 100 * mean, 100 * sd


fig, ax = plt.subplots(figsize=(12, 3.9))
fig.patch.set_facecolor(SURF)
ax.set_axis_off()
ax.set_xlim(0, 1)
n = len(SYSTEMS) + 1
ax.set_ylim(-0.8, n + 1.2)
xs = [0.27 + i * 0.125 for i in range(len(COLS))]
top = n + 0.6
for x, (name, _) in zip(xs, COLS):
    ax.text(x, top, name, ha="center", va="center", fontsize=12, color=INK2, fontweight="bold", linespacing=1.1)
ax.plot([0, 1], [top - 0.6, top - 0.6], color=INK2, lw=1.2)

y = top - 1.1
for g in ["151 labels", "150 labels + threshold"]:
    ax.text(0.01, y, g, ha="left", va="center", fontsize=11.5, color=MUTED, fontstyle="italic")
    y -= 0.75
    rows = [s for s in SYSTEMS if s[2] == g]
    best = {m: max(cell(k, m)[0] for _, k, _ in rows) for _, m in COLS}
    for name, k, _ in rows:
        if k.startswith("lora"):
            ax.add_patch(plt.Rectangle((0, y - 0.4), 1, 0.8, color=BAND, zorder=0, lw=0))
        label = name + (f" (r {R[k]['config']['r']})" if k.startswith("lora") else "")
        ax.text(0.03, y, label, ha="left", va="center", fontsize=12.5, color=INK)
        for x, (_, m) in zip(xs, COLS):
            v, sd = cell(k, m)
            s = f"{v:.1f}" + (f" ± {sd:.1f}" if sd is not None else "")
            ax.text(x, y, s, ha="center", va="center", fontsize=12, color=INK,
                    fontweight="bold" if abs(v - best[m]) < 1e-9 else "normal")
        y -= 0.8
    last = g != "151 labels"
    ax.plot([0, 1], [y + 0.4, y + 0.4], color=INK2 if last else GRID, lw=1.2 if last else 0.8)
    y -= 0.15
ax.text(0, y - 0.1, "Test set. LoRA: mean ± std over 3 seeds. 151: refuse if out_of_scope is the top label. "
        "150: refuse if top-intent probability < τ (τ chosen on val by OOS F1).",
        ha="left", va="center", fontsize=9.5, color=MUTED)

os.makedirs(os.path.join(HERE, "results", "common"), exist_ok=True)
out = os.path.join(HERE, "results", "common", "fig_llm_results_table.png")
fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.15)
print(out)
