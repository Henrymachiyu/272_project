#!/usr/bin/env python3
"""Shared evaluation for every model in the project (TF-IDF, RoBERTa, LLMs).

Input: one directory per model containing val.npz and test.npz with
    labels   (C,)   candidate labels; include 'oos' for a 151-way model
    gold     (N,)   gold label strings ('oos' = out-of-scope)
  and either
    logprobs (N, C) unnormalised scores / log-probs   (softmaxed here), or
    probs    (N, C) probabilities.
  (TF-IDF / RoBERTa: save predict_proba / softmax as `probs` with the same keys.)

Two decision rules are evaluated per model:
  argmax     (151-way models only) predict the top class, which may be oos
  threshold  predict the top in-scope intent unless its probability (MSP) is
             below tau -> oos. For 151-way models an argmax=oos stays oos.
             tau is chosen on val (default criterion: (in-scope acc + OOS recall)/2,
             which does not depend on the val/test OOS ratio), then fixed on test.

Metrics: in-scope accuracy, in-scope macro-F1 (150 intents), OOS precision /
recall / F1, overall 151-way accuracy, accuracy-vs-coverage curve and AURC.

Confidence scores (see Run.confidence_scores); every one is reported with
AUROC / AUPR(OOS) / FPR@95%-in-scope-TPR and with its own val-tuned threshold:
  msp          max in-scope probability of the label-set softmax   (all models)
  margin       top-1 minus top-2 in-scope probability              (all models)
  neg_entropy  minus entropy of the label distribution             (150-way models)
  not_oos      1 - P(oos)                                          (151-way models)
  max_logprob  un-normalised log P(top intent | prompt)            (LLMs, needs logprobs)
  label_mass   log sum_c P(intent_c | prompt): how much of the LM's
               probability goes to *any* valid intent              (LLMs, needs logprobs)

    python evaluate.py runs/zs_151 runs/zs_150 runs/lora_151 runs/lora_150 --out results
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OOS = "oos"


def load_run(path, split):
    d = np.load(os.path.join(path, f"{split}.npz"), allow_pickle=False)
    labels = np.array([str(x) for x in d["labels"]])
    raw = None
    if "probs" in d:
        P = d["probs"].astype(np.float64)
        P = P / P.sum(1, keepdims=True)
    else:
        raw = d["logprobs"].astype(np.float64)
        z = raw.copy()
        z -= z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
    gold = np.array([str(x) for x in d["gold"]])
    texts = np.array([str(x) for x in d["texts"]]) if "texts" in d else None
    return labels, P, raw, gold, texts


class Run:
    def __init__(self, path, split):
        self.labels, self.P, self.raw, self.gold, self.texts = load_run(path, split)
        self.has_oos = OOS in self.labels
        self.in_idx = np.array([i for i, l in enumerate(self.labels) if l != OOS])
        Pin = self.P[:, self.in_idx]
        self.msp = Pin.max(1)
        self.in_pred = self.labels[self.in_idx][Pin.argmax(1)]
        self.argmax_pred = self.labels[self.P.argmax(1)]
        self.p_oos = self.P[:, list(self.labels).index(OOS)] if self.has_oos else None
        self.scores = self.confidence_scores()

    def confidence_scores(self):
        """In-scope confidence scores (higher = more confident the input is a
        supported intent).  P is the softmax over the candidate label set of
        s_c = log P(label_c | prompt); raw = the s_c themselves (LLMs only)."""
        Pin = np.sort(self.P[:, self.in_idx], 1)
        sc = {"msp": Pin[:, -1],                                   # max softmax prob (in-scope)
              "margin": Pin[:, -1] - Pin[:, -2]}                   # top-1 minus top-2
        if not self.has_oos:
            # only for 150-way: with an oos class, a confident oos prediction has
            # low entropy, so entropy is not an in-scope confidence there
            sc["neg_entropy"] = (self.P * np.log(np.clip(self.P, 1e-300, 1))).sum(1)
        else:
            sc["not_oos"] = 1.0 - self.p_oos                       # 1 - P(out_of_scope)
        if self.raw is not None:
            r = self.raw[:, self.in_idx]
            m = r.max(1)
            sc["max_logprob"] = m                                  # un-normalised log P(top label)
            sc["label_mass"] = m + np.log(np.exp(r - m[:, None]).sum(1))  # log P(answer is any intent)
        return sc

    def predict(self, rule, tau=None, score="msp"):
        if rule == "argmax":
            return self.argmax_pred.copy()
        pred = np.where(self.scores[score] >= tau, self.in_pred, OOS)
        if self.has_oos:
            pred[self.argmax_pred == OOS] = OOS
        return pred


def metrics(gold, pred, intents):
    isin, isoos = gold != OOS, gold == OOS
    p_oos = pred == OOS
    tp = (p_oos & isoos).sum()
    prec = tp / max(p_oos.sum(), 1)
    rec = tp / max(isoos.sum(), 1)
    return {
        "in_acc": float((pred == gold)[isin].mean()),
        "in_macro_f1": float(f1_score(gold, pred, labels=intents, average="macro", zero_division=0)),
        "oos_precision": float(prec),
        "oos_recall": float(rec),
        "oos_f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "overall_acc": float((pred == gold).mean()),
    }


def criterion(m, name):
    return {"balanced": 0.5 * (m["in_acc"] + m["oos_recall"]),
            "oos_f1": m["oos_f1"],
            "overall_acc": m["overall_acc"]}[name]


def pick_tau(run, intents, crit, score="msp"):
    x = run.scores[score]
    cands = np.unique(np.concatenate([[-np.inf], np.quantile(x, np.linspace(0, 1, 401))]))
    best = max(cands, key=lambda t: criterion(
        metrics(run.gold, run.predict("threshold", t, score), intents), crit))
    return float(best)


def ood_metrics(gold, conf):
    """OOS detection quality of a confidence score (OOS = positive class, score = -conf)."""
    y = gold == OOS
    t95 = np.quantile(conf[~y], 0.05)          # accept 95% of in-scope inputs
    return {"auroc": float(roc_auc_score(y, -conf)),
            "aupr_oos": float(average_precision_score(y, -conf)),
            "fpr95": float((conf[y] >= t95).mean())}   # share of OOS still accepted


def coverage_curve(run, score="msp"):
    """Answer the most confident fraction first; an answered OOS example is an error."""
    order = np.argsort(-run.scores[score], kind="stable")
    correct = (run.in_pred == run.gold)[order]
    n = len(correct)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    return cov, acc, float(np.mean(1 - acc))   # AURC = mean selective risk


def confusions(gold, pred, k=25):
    from collections import Counter
    c = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [{"gold": g, "pred": p, "count": n} for (g, p), n in c.most_common(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--criterion", default="balanced", choices=["balanced", "oos_f1", "overall_acc"])
    ap.add_argument("--score", default="msp", help="confidence score used by the threshold rule")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    allres, curves = {}, {}
    for path in args.runs:
        name = os.path.basename(os.path.normpath(path))
        val, test = Run(path, "val"), Run(path, "test")
        intents = sorted(l for l in test.labels if l != OOS)
        res = {"n_test": int(len(test.gold)), "n_classes": int(len(test.labels))}

        if test.has_oos:
            res["argmax"] = metrics(test.gold, test.predict("argmax"), intents)
        # every confidence score: OOS-detection quality + threshold rule tuned on val
        res["scores"] = {}
        for sname in test.scores:
            t = pick_tau(val, intents, args.criterion, sname)
            res["scores"][sname] = {**ood_metrics(test.gold, test.scores[sname]), "tau": t,
                                    **metrics(test.gold, test.predict("threshold", t, sname), intents)}
        sc = args.score if args.score in test.scores else "msp"
        res["threshold"] = {"score": sc, **{k: res["scores"][sc][k] for k in
                            ["tau", "in_acc", "in_macro_f1", "oos_precision", "oos_recall", "oos_f1", "overall_acc"]}}
        res["auroc"] = res["scores"][sc]["auroc"]
        cov, acc, aurc = coverage_curve(test, sc)
        res["aurc"] = aurc
        main_pred = test.predict("argmax") if test.has_oos else test.predict("threshold", res["threshold"]["tau"], sc)
        res["top_confusions"] = confusions(test.gold, main_pred)
        allres[name], curves[name] = res, (cov, acc)

        # per-example predictions for error analysis / the demo
        with open(os.path.join(args.out, f"{name}_test_predictions.tsv"), "w") as f:
            f.write("text\tgold\tpred\tmsp\n")
            texts = test.texts if test.texts is not None else [""] * len(test.gold)
            for t, g, p, s in zip(texts, test.gold, main_pred, test.msp):
                f.write(f"{t}\t{g}\t{p}\t{s:.4f}\n")
        # (msp column = max in-scope softmax probability)

    json.dump(allres, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)

    # markdown table
    cols = ["in_acc", "in_macro_f1", "oos_precision", "oos_recall", "oos_f1", "overall_acc"]
    lines = ["| model | rule | tau | in-scope acc | in-scope macro-F1 | OOS P | OOS R | OOS F1 | overall acc | AUROC | AURC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in allres.items():
        for rule in ("argmax", "threshold"):
            if rule not in r:
                continue
            m = r[rule]
            tau = f"{m['tau']:.3f}" if "tau" in m else "-"
            lines.append(f"| {name} | {rule} | {tau} | " + " | ".join(f"{100 * m[c]:.1f}" for c in cols)
                         + f" | {100 * r['auroc']:.1f} | {100 * r['aurc']:.1f} |")
    lines += ["", f"Threshold rule / AUROC / AURC use score = {args.score}; tau tuned on val ({args.criterion}).", "",
              "Confidence-score ablation (test; tau tuned on val per score)", "",
              "| model | score | AUROC | AUPR(OOS) | FPR@95 | in-scope acc | OOS R | OOS F1 | overall acc |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, r in allres.items():
        for sname, m in r["scores"].items():
            lines.append(f"| {name} | {sname} | " + " | ".join(f"{100 * m[c]:.1f}" for c in
                         ["auroc", "aupr_oos", "fpr95", "in_acc", "oos_recall", "oos_f1", "overall_acc"]) + " |")
    table = "\n".join(lines)
    open(os.path.join(args.out, "table.md"), "w").write(table + "\n")
    print(table)

    # accuracy-coverage plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (cov, acc) in curves.items():
        ax.plot(cov, acc, label=name)
    ax.set_xlabel("coverage (fraction of test inputs answered)")
    ax.set_ylabel("accuracy on answered inputs")
    ax.set_title("Accuracy vs. coverage (CLINC150 test)")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "coverage.png"), dpi=150)


if __name__ == "__main__":
    main()
