# figure + error tables for the two roberta runs of one seed
#   python analyze.py [tag]     tag defaults to s0, i.e. runs/m151_s0 and runs/m150_s0
# needs the test_probs.npy files written by roberta_clinc.py
# writes figs/roberta_cutoff_curves_<tag>.png and error_analysis_<tag>.md
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
N_INTENTS = 150


def load_run(run_dir):
    probs = np.load(os.path.join(ROOT, run_dir, "test_probs.npy"))
    labels = json.load(open(os.path.join(ROOT, run_dir, "labels.json")))
    results = json.load(open(os.path.join(ROOT, run_dir, "results.json")))
    return probs, labels, results


def refused(probs, conf, cutoff, mode):
    # same rule as roberta_clinc.py: top intent prob below the cutoff, or (mode 151) the oos class wins
    refuse = conf < cutoff
    if mode == "151":
        refuse = refuse | (probs.argmax(1) == N_INTENTS)
    return refuse


def count_errors(gold, pred):
    wrong_intent = Counter()   # (gold intent, predicted intent)
    oos_accepted = Counter()   # oos sentence answered with this intent
    in_refused = Counter()     # in-scope sentence of this intent refused
    for g, p in zip(gold, pred):
        if g != "oos" and p != "oos" and g != p:
            wrong_intent[(g, p)] += 1
        elif g == "oos" and p != "oos":
            oos_accepted[p] += 1
        elif g != "oos" and p == "oos":
            in_refused[g] += 1

    return wrong_intent, oos_accepted, in_refused


def join_counts(items):
    return "; ".join(f"{name} ({n})" for name, n in items)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "s0"
    runs = [
        ("RoBERTa 151-class", "151", f"runs/m151_{tag}"),
        ("RoBERTa 150-class + cutoff", "150", f"runs/m150_{tag}"),
    ]

    data = json.load(open(os.path.join(ROOT, "data", "data_oos_plus.json")))
    test = data["test"] + data["oos_test"]
    gold = np.array([label for _, label in test], dtype=object)
    in_scope = gold != "oos"
    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)

    fig, (ax_cov, ax_cut) = plt.subplots(1, 2, figsize=(11, 4))
    md = []

    for name, mode, run_dir in runs:
        probs, labels, results = load_run(run_dir)
        intents = np.array(labels[:N_INTENTS], dtype=object)
        conf = probs[:, :N_INTENTS].max(1)
        top_intent = intents[probs[:, :N_INTENTS].argmax(1)]
        correct = in_scope & (top_intent == gold)
        val_cutoff = results["test_val_cutoff"]["cutoff"]

        # left: answer the most confident inputs first; answering an oos input counts as wrong
        order = np.argsort(-conf)
        n_answered = np.arange(1, len(order) + 1)
        coverage = n_answered / len(order)
        acc_answered = np.cumsum(correct[order]) / n_answered
        ax_cov.plot(coverage, acc_answered, label=name)

        # right: in-scope accuracy and oos recall for every cutoff
        cutoffs = np.linspace(0, 1, 101)
        in_acc = []
        oos_rec = []
        for c in cutoffs:
            refuse = refused(probs, conf, c, mode)
            in_acc.append((correct & ~refuse)[in_scope].mean())
            oos_rec.append(refuse[~in_scope].mean())

        line, = ax_cut.plot(cutoffs, in_acc, label=f"{name}: in-scope acc")
        ax_cut.plot(cutoffs, oos_rec, "--", color=line.get_color(), label=f"{name}: OOS recall")
        ax_cut.axvline(val_cutoff, color=line.get_color(), lw=0.8, alpha=0.6)

        # error tables at the cutoff picked on val
        pred = top_intent.copy()
        pred[refused(probs, conf, val_cutoff, mode)] = "oos"
        wrong_intent, oos_accepted, in_refused = count_errors(gold, pred)

        pairs = [f"{a} -> {b} ({k})" for (a, b), k in wrong_intent.most_common(10)]
        md.append(f"### {name} (cutoff {val_cutoff:.2f})\n")
        md.append(f"- in-scope wrong intent: {sum(wrong_intent.values())} | "
                  f"in-scope refused: {sum(in_refused.values())} | OOS accepted: {sum(oos_accepted.values())}\n")
        md.append("- top confused pairs (gold -> pred): " + "; ".join(pairs))
        md.append("- OOS accepted as: " + join_counts(oos_accepted.most_common(10)))
        md.append("- in-scope intents refused most: " + join_counts(in_refused.most_common(10)) + "\n")

    ax_cov.axvline(in_scope.mean(), color="gray", ls=":", lw=1, label="all in-scope answered, all OOS refused")
    ax_cov.set_xlabel("coverage (fraction of test inputs answered)")
    ax_cov.set_ylabel("accuracy on answered inputs")
    ax_cov.set_title("Answer the most confident inputs first")
    ax_cov.legend(fontsize=8)
    ax_cov.set_ylim(0.75, 1.0)
    ax_cov.grid(alpha=0.3)

    ax_cut.set_xlabel("cutoff on top in-scope probability")
    ax_cut.set_title("Effect of the cutoff (vertical line = picked on val)")
    ax_cut.legend(fontsize=7)
    ax_cut.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "figs", f"roberta_cutoff_curves_{tag}.png"), dpi=150)

    txt = "\n".join(md)
    open(os.path.join(ROOT, f"error_analysis_{tag}.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
