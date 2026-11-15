# mean ± std over seeds 0-2 for both modes
import json
import os

import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
seeds = [0, 1, 2]
keys = ["in_scope_acc", "in_scope_macro_f1", "oos_precision", "oos_recall", "oos_f1", "oos_auroc", "overall_acc_151"]
rows = [
    ("151-class (argmax)", "151", "test_no_cutoff"),
    ("151-class + val cutoff", "151", "test_val_cutoff"),
    ("150-class, no cutoff", "150", "test_no_cutoff"),
    ("150-class + val cutoff", "150", "test_val_cutoff"),
]

res = {}
for m in ["151", "150"]:
    for s in seeds:
        path = os.path.join(root, f"runs/m{m}_s{s}/results.json")
        res[(m, s)] = json.load(open(path))

out = []
out.append("| method | " + " | ".join(keys) + " | cutoffs | best epochs |")
out.append("|" + "---|" * (len(keys) + 3))

for name, m, k in rows:
    cells = []
    for key in keys:
        v = np.array([res[(m, s)][k][key] for s in seeds]) * 100
        cells.append(f"{v.mean():.2f} ± {v.std(ddof=1):.2f}")

    cuts = []
    eps = []
    for s in seeds:
        cuts.append(f"{res[(m, s)][k]['cutoff']:.2f}")
        eps.append(str(res[(m, s)]["best_epoch"]))

    out.append(f"| {name} | " + " | ".join(cells) + f" | {', '.join(cuts)} | {', '.join(eps)} |")

# per-seed gap between the two oos methods (same seed)
out.append("\nper-seed (151 + cutoff) minus (150 + cutoff):")
for key in ["in_scope_acc", "oos_recall", "oos_precision", "oos_f1", "oos_auroc"]:
    diffs = []
    for s in seeds:
        a = res[("151", s)]["test_val_cutoff"][key]
        b = res[("150", s)]["test_val_cutoff"][key]
        diffs.append(f"{100 * (a - b):+.2f}")

    out.append(f"- {key}: " + ", ".join(diffs))

txt = "\n".join(out)
open(os.path.join(root, "results_seeds.md"), "w").write(txt + "\n")
print(txt)
