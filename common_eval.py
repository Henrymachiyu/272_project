#!/usr/bin/env python3
"""Evaluate zero-shot and LoRA under the team's shared protocol (same as the RoBERTa report).

  (a) 151 labels: train with oos; refuse iff argmax == oos.      OOS score for AUROC: P(oos)
  (b) 150 labels: in-scope train only; refuse iff MSP < t.       OOS score for AUROC: 1 - MSP
      t in {0, 0.01, ..., 0.99}, chosen on val by OOS F1.
Model selection on val without any OOS threshold: (a) 151-class overall accuracy, (b) in-scope
accuracy. LoRA (lr, r) is selected this way on seed 42, then reported as mean +- std over seeds 42-44.

    python common_eval.py      # writes results/common/{common_report.json, common_table.md}
"""

import itertools
import json
import os

import numpy as np

import evaluate as ev
from sweep import name

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, GRID = [42, 43, 44], {"lr": [1e-4, 2e-4, 4e-4], "r": [8, 16, 32]}
ZS = {"151": "runs/zs_151", "150": "runs/zs_150"}
KEYS = ["in_acc", "oos_recall", "oos_precision", "oos_f1", "in_macro_f1", "auroc", "overall_acc"]
T_GRID = np.round(np.arange(0, 1.0, 0.01), 2)


def evaluate_split(path, v, split, tau=None):
    run = ev.Run(path, split)
    intents = sorted(l for l in run.labels if l != ev.OOS)
    if v == "151":
        pred, conf = run.predict("argmax"), run.scores["not_oos"]     # 1 - P(oos): same AUROC as P(oos)
    else:
        if tau is None:   # first t with the best val OOS F1
            f1 = [ev.metrics(run.gold, run.predict("threshold", t, "msp"), intents)["oos_f1"] for t in T_GRID]
            tau = float(T_GRID[int(np.argmax(f1))])
        pred, conf = run.predict("threshold", tau, "msp"), run.scores["msp"]
    m = ev.metrics(run.gold, pred, intents)
    m.update(ev.ood_metrics(run.gold, conf))
    m["tau"] = tau
    return m, run, pred


def test_eval(path, v):
    tau = None if v == "151" else evaluate_split(path, v, "val")[0]["tau"]
    return evaluate_split(path, v, "test", tau)


def bootstrap(run, pred, conf, B=1000, seed=0):
    intents = sorted(l for l in run.labels if l != ev.OOS)
    rng, n, out = np.random.default_rng(seed), len(run.gold), {k: [] for k in KEYS}
    for _ in range(B):
        i = rng.integers(0, n, n)
        m = ev.metrics(run.gold[i], pred[i], intents)
        m.update(ev.ood_metrics(run.gold[i], conf[i]))
        for k in KEYS:
            out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def breakdown(run, pred):
    g, isin = run.gold, run.gold != ev.OOS
    return {"correct_in": int(((pred == g) & isin).sum()),
            "wrong_intent": int(((pred != g) & isin & (pred != ev.OOS)).sum()),
            "refused_in": int((isin & (pred == ev.OOS)).sum()),
            "correct_oos": int((~isin & (pred == ev.OOS)).sum()),
            "accepted_oos": int((~isin & (pred != ev.OOS)).sum())}


def main():
    os.chdir(HERE)
    report, L = {}, ["| system | config | " + " | ".join(KEYS) + " |", "|---" * (2 + len(KEYS)) + "|"]
    for v in ["151", "150"]:
        # LoRA config selected on val (seed 42) without any OOS threshold
        cands = []
        for lr, r in itertools.product(GRID["lr"], GRID["r"]):
            val = ev.Run(f"runs/sweep/{name(v, lr, r, 42)}", "val")
            isin = val.gold != ev.OOS
            acc = (val.argmax_pred == val.gold).mean() if v == "151" else (val.in_pred == val.gold)[isin].mean()
            cands.append((round(float(acc), 6), lr, r))
        _, lr, r = max(cands)

        # zero-shot: deterministic -> bootstrap CI
        zm, zrun, zpred = test_eval(ZS[v], v)
        ci = bootstrap(zrun, zpred, zrun.scores["not_oos" if v == "151" else "msp"])
        report[f"zs_{v}"] = {"test": zm, "ci95": ci, "breakdown": breakdown(zrun, zpred),
                             "confusions": ev.confusions(zrun.gold, zpred, 15)}
        L.append(f"| zero-shot {v} | - | " + " | ".join(
            f"{100 * zm[k]:.1f} [{100 * ci[k][0]:.1f}, {100 * ci[k][1]:.1f}]" for k in KEYS) + " |")

        per_seed, bds, conf42 = {}, [], None
        for s in SEEDS:
            m, run, pred = test_eval(f"runs/sweep/{name(v, lr, r, s)}", v)
            per_seed[s] = m
            bds.append(breakdown(run, pred))
            if s == 42:
                conf42 = ev.confusions(run.gold, pred, 15)
        agg = {k: (float(np.mean([per_seed[s][k] for s in SEEDS])),
                   float(np.std([per_seed[s][k] for s in SEEDS], ddof=1))) for k in KEYS}
        report[f"lora_{v}"] = {"config": {"lr": lr, "r": r}, "val_grid": cands, "per_seed": per_seed,
                               "mean_std": agg, "breakdown_mean": {k: float(np.mean([b[k] for b in bds]))
                                                                   for k in bds[0]},
                               "confusions_seed42": conf42}
        L.append(f"| LoRA {v} | lr={lr:g}, r={r} | " + " | ".join(
            f"{100 * agg[k][0]:.1f} ± {100 * agg[k][1]:.1f}" for k in KEYS) + " |")

    L += ["", "151: refuse iff argmax = oos; AUROC on P(oos). 150: refuse iff MSP < t, t in {0, 0.01, ..., 0.99} "
          "chosen on val by OOS F1; AUROC on 1 - MSP. LoRA config chosen on val by 151-class accuracy (151) or "
          "in-scope accuracy (150); mean ± std over seeds 42-44. Zero-shot: [95% bootstrap CI]."]
    os.makedirs("results/common", exist_ok=True)
    json.dump(report, open("results/common/common_report.json", "w"), indent=1, default=float)
    open("results/common/common_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
