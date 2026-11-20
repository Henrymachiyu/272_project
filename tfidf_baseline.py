#!/usr/bin/env python3
"""TF-IDF + logistic regression baseline on CLINC150 (OOS-plus), under the team protocol.

  1. Preprocessing (inside the vectorizers, so a saved model takes raw text):
     lowercase, whitespace normalisation, no truncation (utterances are <= 28 words).
  2. Features: word 1-2 gram TF-IDF + char_wb 2-5 gram TF-IDF (min_df=2), sublinear tf.
  3. Classifier: multinomial LogisticRegression; C in {1, 3, 10, 30, 100} chosen on val
     without any OOS threshold, as in common_eval.py:
       151 labels: train + oos_train, C by val 151-way accuracy
       150 labels: in-scope train only, C by val in-scope accuracy
  4. Val/test probabilities are saved as runs/tfidf_{151,150}/{val,test}.npz, the input
     format of evaluate.py, so every model in the project is scored by the same code.
  5. Test metrics come from common_eval.test_eval:
       151: refuse iff argmax = oos                      AUROC on P(oos)
       150: refuse iff MSP < t, t in {0, 0.01, ..., 0.99} chosen on val by OOS F1
                                                         AUROC on 1 - MSP
     TF-IDF is deterministic, so 95% bootstrap CIs are reported (as for zero-shot).

    python download_data.py
    python tfidf_baseline.py --save-models     # ~1 min on a multi-core CPU
    python tfidf_visualize.py

Outputs: results/tfidf/{tfidf_report.json, tfidf_table.md, curves.json},
         results/tfidf/test_predictions.csv (git-ignored), models/tfidf_{151,150}.joblib.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

import common_eval as ce
import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "data_oos_plus.json")
RUNS = {"151": os.path.join(HERE, "runs", "tfidf_151"), "150": os.path.join(HERE, "runs", "tfidf_150")}
OUT = os.path.join(HERE, "results", "tfidf")
MODELS = os.path.join(HERE, "models")
OOS = ev.OOS
C_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]


# ----------------------------------------------------------------------------- data
def load_splits(path=DATA):
    """{split: (texts, labels)}; 'train' includes oos_train, 'train_in' is in-scope only."""
    D = json.load(open(path))
    pack = lambda rows: ([t for t, _ in rows], np.array([lab for _, lab in rows]))
    return {"train_in": pack(D["train"]),
            "train": pack(D["train"] + D["oos_train"]),
            "val": pack(D["val"] + D["oos_val"]),
            "test": pack(D["test"] + D["oos_test"])}


# ---------------------------------------------------------------------------- model
def make_pipeline(C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                                 sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("tfidf", feats), ("lr", LogisticRegression(C=C, max_iter=3000))])


def fit(texts, labels, C):
    return make_pipeline(C).fit(texts, labels)


def select_C(S):
    """Fit both label settings over the C grid in parallel; pick C on val (no threshold)."""
    jobs = [(v, C) for v in ("151", "150") for C in C_GRID]
    t0 = time.time()
    fitted = Parallel(n_jobs=len(jobs))(
        delayed(fit)(*(S["train"] if v == "151" else S["train_in"]), C) for v, C in jobs)
    print(f"fit {len(jobs)} models in {time.time() - t0:.0f}s")
    Xv, yv = S["val"]
    isin = yv != OOS
    grid, best = {"151": [], "150": []}, {}
    for (v, C), model in zip(jobs, fitted):
        pv = model.predict(Xv)
        acc = float((pv == yv).mean() if v == "151" else (pv == yv)[isin].mean())
        grid[v].append({"C": C, "val_acc_151" if v == "151" else "val_in_acc": acc})
        print(f"  {v} labels  C={C:<5g} val {'151-way' if v == '151' else 'in-scope'} acc = {acc:.4f}")
        if v not in best or acc > best[v][0]:
            best[v] = (acc, C, model)
    return {v: (C, m) for v, (_, C, m) in best.items()}, grid


def export_run(model, S, path):
    """Save val/test probabilities in the evaluate.py format."""
    os.makedirs(path, exist_ok=True)
    for split in ("val", "test"):
        X, y = S[split]
        np.savez_compressed(os.path.join(path, f"{split}.npz"),
                            labels=np.array(model.classes_, dtype=str), gold=y.astype(str),
                            probs=model.predict_proba(X), texts=np.array(X, dtype=str))


# --------------------------------------------------------------------------- curves
def threshold_curve(run, n=201):
    """Sweep the MSP cutoff on test (151: argmax = oos always stays a refusal)."""
    intents = sorted(l for l in run.labels if l != OOS)
    out = []
    for t in np.linspace(0, 1, n):
        m = ev.metrics(run.gold, run.predict("threshold", t, "msp"), intents)
        out.append({"t": round(float(t), 3), "in_acc": m["in_acc"], "oos_recall": m["oos_recall"]})
    return out


def coverage_points(run, n=200):
    cov, acc, aurc = ev.coverage_curve(run, "msp")
    idx = np.unique(np.linspace(0, len(cov) - 1, n).astype(int))
    return {"coverage": cov[idx].tolist(), "selective_acc": acc[idx].tolist(), "aurc": aurc}


def intent_confusions(gold, pred, k=15):
    """Most frequent in-scope -> other in-scope intent errors (oos on neither side)."""
    keep = (gold != OOS) & (pred != OOS)
    return ev.confusions(gold[keep], pred[keep], k)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-models", action="store_true", help="write models/tfidf_{151,150}.joblib")
    args = ap.parse_args()
    if not os.path.exists(DATA):
        raise SystemExit("data/data_oos_plus.json not found: run `python download_data.py` first")
    os.makedirs(OUT, exist_ok=True)

    S = load_splits()
    best, grid = select_C(S)
    for v, (C, model) in best.items():
        export_run(model, S, RUNS[v])

    report, curves, preds = {}, {}, {}
    L = ["| system | config | " + " | ".join(ce.KEYS) + " |", "|---" * (2 + len(ce.KEYS)) + "|"]
    for v in ("151", "150"):
        m, run, pred = ce.test_eval(RUNS[v], v)
        conf = run.scores["not_oos" if v == "151" else "msp"]
        ci = ce.bootstrap(run, pred, conf)
        C = best[v][0]
        report[f"tfidf_{v}"] = {"config": {"C": C}, "val_grid": grid[v], "test": m, "ci95": ci,
                                "breakdown": ce.breakdown(run, pred),
                                "confusions": ev.confusions(run.gold, pred, 15),
                                "intent_confusions": intent_confusions(run.gold, pred, 15)}
        cfg = f"C={C:g}" + ("" if v == "151" else f", t={m['tau']:.2f}")
        L.append(f"| TF-IDF {v} | {cfg} | " + " | ".join(
            f"{100 * m[k]:.1f} [{100 * ci[k][0]:.1f}, {100 * ci[k][1]:.1f}]" for k in ce.KEYS) + " |")
        curves[v] = {"threshold": threshold_curve(run), **coverage_points(run),
                     "operating_point": {"in_acc": m["in_acc"], "oos_recall": m["oos_recall"],
                                         "tau": m["tau"]}}
        preds[v] = (run, pred)

    L += ["", "151: refuse iff argmax = oos; AUROC on P(oos). 150: refuse iff MSP < t, t in {0, 0.01, ..., 0.99} "
          "chosen on val by OOS F1; AUROC on 1 - MSP. C chosen on val by 151-way accuracy (151) or in-scope "
          "accuracy (150). Deterministic model: [95% bootstrap CI over test queries]."]
    json.dump(report, open(os.path.join(OUT, "tfidf_report.json"), "w"), indent=1, default=float)
    json.dump(curves, open(os.path.join(OUT, "curves.json"), "w"))
    open(os.path.join(OUT, "tfidf_table.md"), "w").write("\n".join(L) + "\n")

    (r151, p151), (r150, p150) = preds["151"], preds["150"]
    with open(os.path.join(OUT, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "gold", "pred_151", "p_oos_151", "pred_150", "msp_150"])
        for row in zip(S["test"][0], r151.gold, p151, r151.p_oos, p150, r150.msp):
            w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}", row[4], f"{row[5]:.4f}"])

    if args.save_models:
        import joblib
        os.makedirs(MODELS, exist_ok=True)
        joblib.dump({"model": best["151"][1], "rule": "argmax"},
                    os.path.join(MODELS, "tfidf_151.joblib"), compress=3)
        joblib.dump({"model": best["150"][1], "rule": "msp", "tau": report["tfidf_150"]["test"]["tau"]},
                    os.path.join(MODELS, "tfidf_150.joblib"), compress=3)

    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
