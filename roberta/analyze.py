# plots + error tables for the two roberta runs
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(__file__))

tag = "s0"
if len(sys.argv) > 1:
    tag = sys.argv[1]

runs = {
    "RoBERTa 151-class": f"runs/m151_{tag}",
    "RoBERTa 150-class + cutoff": f"runs/m150_{tag}",
}

d = json.load(open(os.path.join(root, "data", "data_oos_plus.json")))
test = d["test"] + d["oos_test"]
gold = np.array([y for _, y in test], dtype=object)
ins = gold != "oos"
os.makedirs(os.path.join(root, "figs"), exist_ok=True)


def load(r):
    p = np.load(os.path.join(root, r, "test_probs.npy"))
    labels = json.load(open(os.path.join(root, r, "labels.json")))
    res = json.load(open(os.path.join(root, r, "results.json")))
    return p, labels, res


def join_counts(items):
    parts = []
    for a, n in items:
        parts.append(f"{a} ({n})")
    return "; ".join(parts)


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
md = []

for name, r in runs.items():
    p, labels, res = load(r)
    intents = np.array(labels[:150], dtype=object)
    conf = p[:, :150].max(1)
    top = intents[p[:, :150].argmax(1)]
    right = ins & (top == gold)
    val_cut = res["test_val_cutoff"]["cutoff"]

    # accuracy vs coverage: answer the most confident inputs first, answering an oos input is wrong
    order = np.argsort(-conf)
    n = np.arange(1, len(order) + 1)
    cov = n / len(order)
    acc = np.cumsum(right[order]) / n
    ax[0].plot(cov, acc, label=name)

    # in-scope acc and oos recall vs cutoff
    cuts = np.linspace(0, 1, 101)
    in_acc = []
    oos_rec = []
    for c in cuts:
        refuse = conf < c
        if "151" in name:
            refuse = refuse | (p.argmax(1) == 150)

        answered_right = right & ~refuse
        in_acc.append(answered_right[ins].mean())
        oos_rec.append(refuse[~ins].mean())

    l, = ax[1].plot(cuts, in_acc, label=f"{name}: in-scope acc")
    ax[1].plot(cuts, oos_rec, "--", color=l.get_color(), label=f"{name}: OOS recall")
    ax[1].axvline(val_cut, color=l.get_color(), lw=0.8, alpha=0.6)

    # error tables at the val-chosen cutoff
    pred = top.copy()
    refuse = conf < val_cut
    if "151" in name:
        refuse |= p.argmax(1) == 150

    pred[refuse] = "oos"

    confused = Counter()
    accepted = Counter()
    refused = Counter()
    for g, q in zip(gold, pred):
        if g != "oos" and q != "oos" and g != q:
            confused[(g, q)] += 1
        elif g == "oos" and q != "oos":
            accepted[q] += 1
        elif g != "oos" and q == "oos":
            refused[g] += 1

    pairs = []
    for (a, b), k in confused.most_common(10):
        pairs.append(f"{a} -> {b} ({k})")

    n_wrong = sum(confused.values())
    n_refused = sum(refused.values())
    n_accepted = sum(accepted.values())
    md.append(f"### {name} (cutoff {val_cut:.2f})\n")
    md.append(f"- in-scope wrong intent: {n_wrong} | in-scope refused: {n_refused} | OOS accepted: {n_accepted}\n")
    md.append("- top confused pairs (gold -> pred): " + "; ".join(pairs))
    md.append("- OOS accepted as: " + join_counts(accepted.most_common(10)))
    md.append("- in-scope intents refused most: " + join_counts(refused.most_common(10)) + "\n")

ax[0].axvline(ins.mean(), color="gray", ls=":", lw=1, label="all in-scope answered, all OOS refused")
ax[0].set_xlabel("coverage (fraction of test inputs answered)")
ax[0].set_ylabel("accuracy on answered inputs")
ax[0].set_title("Answer the most confident inputs first")
ax[0].legend(fontsize=8)
ax[0].set_ylim(0.75, 1.0)
ax[0].grid(alpha=0.3)

ax[1].set_xlabel("cutoff on top in-scope probability")
ax[1].set_title("Effect of the cutoff (vertical line = picked on val)")
ax[1].legend(fontsize=7)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(root, "figs", f"roberta_cutoff_curves_{tag}.png"), dpi=150)

txt = "\n".join(md)
open(os.path.join(root, f"error_analysis_{tag}.md"), "w").write(txt + "\n")
print(txt)
