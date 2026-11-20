# mean ± std over seeds 0-2 for both modes -> results_seeds.md
#   python summarize_seeds.py   (needs runs/m151_s{0,1,2} and runs/m150_s{0,1,2})
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
METRICS = ["in_scope_acc", "in_scope_macro_f1", "oos_precision", "oos_recall", "oos_f1", "oos_auroc",
           "overall_acc_151"]
# (row name, mode, which result in results.json)
ROWS = [
    ("151-class (argmax)", "151", "test_no_cutoff"),
    ("151-class + val cutoff", "151", "test_val_cutoff"),
    ("150-class, no cutoff", "150", "test_no_cutoff"),
    ("150-class + val cutoff", "150", "test_val_cutoff"),
]


def main():
    results = {}
    for mode in ["151", "150"]:
        for seed in SEEDS:
            path = os.path.join(ROOT, f"runs/m{mode}_s{seed}/results.json")
            results[(mode, seed)] = json.load(open(path))

    out = ["| method | " + " | ".join(METRICS) + " | cutoffs | best epochs |",
           "|" + "---|" * (len(METRICS) + 3)]

    for name, mode, key in ROWS:
        runs = [results[(mode, seed)] for seed in SEEDS]
        cells = []
        for metric in METRICS:
            v = np.array([r[key][metric] for r in runs]) * 100
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1):.2f}")

        cutoffs = ", ".join(f"{r[key]['cutoff']:.2f}" for r in runs)
        epochs = ", ".join(str(r["best_epoch"]) for r in runs)
        out.append(f"| {name} | " + " | ".join(cells) + f" | {cutoffs} | {epochs} |")

    # the two oos methods compared within the same seed
    out.append("\nper-seed (151 + cutoff) minus (150 + cutoff):")
    for metric in ["in_scope_acc", "oos_recall", "oos_precision", "oos_f1", "oos_auroc"]:
        diffs = []
        for seed in SEEDS:
            a = results[("151", seed)]["test_val_cutoff"][metric]
            b = results[("150", seed)]["test_val_cutoff"][metric]
            diffs.append(f"{100 * (a - b):+.2f}")

        out.append(f"- {metric}: " + ", ".join(diffs))

    txt = "\n".join(out)
    open(os.path.join(ROOT, "results_seeds.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
