#!/usr/bin/env python3
"""Figures for the TF-IDF baseline (run tfidf_baseline.py first).

    python tfidf_visualize.py
writes results/tfidf/
  fig_tfidf_tradeoff.png   left: in-scope accuracy vs OOS recall as the MSP cutoff moves,
                           with the protocol operating points; right: accuracy on the
                           inputs answered vs coverage (answer the most confident first)
  fig_tfidf_errors.png     test errors by type (wrong intent / in-scope refused / OOS accepted)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "tfidf")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COLOR = {"151": "#2a78d6", "150": "#eb6834"}
NAME = {"151": "151 labels (oos as a class)", "150": "150 labels + MSP cutoff"}
CATS = [("wrong intent", "wrong_intent", "#2a78d6"), ("in-scope refused", "refused_in", "#eb6834"),
        ("OOS accepted", "accepted_oos", "#1baf7a")]   # same categories/colours as error_fig.py

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def tradeoff(curves):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for v in ("151", "150"):
        c, col = curves[v], COLOR[v]
        op = c["operating_point"]
        x, y = op["oos_recall"] * 100, op["in_acc"] * 100
        lab = "argmax" if op["tau"] is None else f"t = {op['tau']:.2f}"
        a1.plot([p["oos_recall"] * 100 for p in c["threshold"]], [p["in_acc"] * 100 for p in c["threshold"]],
                color=col, lw=2, label=f"{NAME[v]}\n  {lab}: {y:.1f}% in-scope acc, {x:.1f}% OOS recall")
        a1.plot(x, y, "o", ms=9, color=col, mec=SURF, mew=2, zorder=5)
        a1.annotate(lab, (x, y), xytext=(6, 8) if v == "151" else (-6, 8), textcoords="offset points",
                    ha="left" if v == "151" else "right", fontsize=9, color=INK2)
        a2.plot([p * 100 for p in c["coverage"]], [p * 100 for p in c["selective_acc"]], color=col, lw=2,
                label=f"{NAME[v]} (AURC {c['aurc']:.3f})")
    a1.set_ylim(50, 97)
    a1.set_xlim(0, 101)
    a1.set_xlabel("OOS recall on test (%)")
    a1.set_ylabel("in-scope accuracy on test (%)")
    a1.set_title("Refusing more OOS costs in-scope accuracy", loc="left")
    a1.legend(loc="lower left")
    a2.set_xlabel("coverage: test inputs answered (%)")
    a2.set_ylabel("accuracy on answered inputs (%)")
    a2.set_title("Answering fewer inputs raises accuracy", loc="left")
    a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tfidf_tradeoff.png"), dpi=200)
    plt.close(fig)


def errors(report):
    rows = [("TF-IDF, 151", "tfidf_151"), ("TF-IDF, 150 + t", "tfidf_150")]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for i, (label, key) in enumerate(rows):
        bd, left = report[key]["breakdown"], 0
        for name, k, col in CATS:
            n = bd[k]
            ax.barh(i, n, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2,
                    label=name if i == 0 else None)
            if n >= 60:
                ax.text(left + n / 2, i, f"{n:,}", ha="center", va="center", color="white", fontsize=10)
            left += n
        ax.text(left + 15, i, f"{left:,} errors / 5,500", va="center", color=INK2, fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(sum(report[k]["breakdown"][c] for _, c, _ in CATS) for _, k in rows) * 1.25)
    ax.set_xlabel("test queries")
    ax.grid(axis="y", visible=False)
    ax.set_title("TF-IDF test errors by type", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tfidf_errors.png"), dpi=200)
    plt.close(fig)


def main():
    curves = json.load(open(os.path.join(OUT, "curves.json")))
    report = json.load(open(os.path.join(OUT, "tfidf_report.json")))
    tradeoff(curves)
    errors(report)
    print(f"wrote {OUT}/fig_tfidf_tradeoff.png and fig_tfidf_errors.png")


if __name__ == "__main__":
    main()
