#!/usr/bin/env python3
"""Hyper-parameter protocol for the LLM experiments.

Data roles:  train -> fit LoRA weights
             val   -> early stopping (val loss picks the epoch), LoRA grid point,
                      confidence score, and cutoff tau
             test  -> touched once, only for the configurations selected on val

Stages
  python sweep.py grid   --gpus 0,1,2,3,4,5,6,7   # train every grid point, score val only
  python sweep.py select                          # choose (config, score) per variant on val
  python sweep.py final  --gpus 0,1,2,3,4,5,6,7   # selected config x 3 seeds, score val+test,
                                                  # report test mean +- std (and zero-shot)
Every job is skipped if its outputs exist, so stages can be re-run / resumed.

Selection criterion on val (same one evaluate.py uses for tau):
    J = max_tau  1/2 * (in-scope accuracy + OOS recall)
ties broken by val AUROC of the score.
"""

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402

PY = sys.executable
VARIANTS = ["151", "150"]
GRID = {"lr": [1e-4, 2e-4, 4e-4], "r": [8, 16, 32]}
EPOCHS = 3
EFFECTIVE_BS = 64          # 16 per step x 4 accumulation on one GPU
SEL_SEED = 42
FINAL_SEEDS = [42, 43, 44]
ZEROSHOT = {"151": "runs/zs_151", "150": "runs/zs_150"}
CRIT = "balanced"


def name(v, lr, r, seed):
    return f"{v}_lr{lr:g}_r{r}_s{seed}"


def job_cmds(v, lr, r, seed, splits, gpu):
    n = name(v, lr, r, seed)
    ck, out = f"ckpt/sweep/{n}", f"runs/sweep/{n}"
    cmds = []
    if not os.path.exists(f"{ck}/final/adapter_model.safetensors"):
        cmds.append([PY, "train_lora.py", "--variant", v, "--out", ck, "--lr", str(lr), "--r", str(r),
                     "--alpha", str(2 * r), "--epochs", str(EPOCHS), "--seed", str(seed),
                     "--bs", "16", "--grad_accum", str(EFFECTIVE_BS // 16)])
    todo = [s for s in splits if not os.path.exists(f"{out}/{s}.npz")]
    if todo:
        cmds.append([PY, "score_llm.py", "--style", "lora", "--adapter", f"{ck}/final", "--out", out,
                     "--splits", ",".join(todo), "--gpus", gpu, "--bs", "32"]
                    + (["--oos_class"] if v == "151" else []))
    return n, cmds


def run_pool(jobs, gpus):
    """jobs: list of (v, lr, r, seed, splits). One job per GPU at a time."""
    os.makedirs("logs/sweep", exist_ok=True)
    free = queue.Queue()
    for g in gpus:
        free.put(g)
    failed = []

    def worker(job):
        g = free.get()
        n, cmds = job_cmds(*job, gpu=g)
        try:
            for c in cmds:
                print(f"[gpu {g}] {n}: {' '.join(c[1:3])}", flush=True)
                with open(f"logs/sweep/{n}.log", "a") as f:
                    rc = subprocess.call(c, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=g,
                                                  TOKENIZERS_PARALLELISM="false"))
                if rc:
                    failed.append(n)
                    print(f"[gpu {g}] {n} FAILED (see logs/sweep/{n}.log)", flush=True)
                    return
            print(f"[gpu {g}] {n} done", flush=True)
        finally:
            free.put(g)

    threads = []
    for job in jobs:
        t = threading.Thread(target=worker, args=(job,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if failed:
        sys.exit(f"failed: {failed}")


# ----------------------------------------------------------------------------
def val_select(path):
    """All (score -> val stats) for one run directory."""
    val = ev.Run(path, "val")
    intents = sorted(l for l in val.labels if l != ev.OOS)
    out = {}
    for s in val.scores:
        tau = ev.pick_tau(val, intents, CRIT, s)
        m = ev.metrics(val.gold, val.predict("threshold", tau, s), intents)
        out[s] = {"J": ev.criterion(m, CRIT), "auroc": ev.ood_metrics(val.gold, val.scores[s])["auroc"],
                  "tau": tau, "in_acc": m["in_acc"], "oos_recall": m["oos_recall"]}
    return out


def stage_select(args):
    rows, selected = [], {}
    for v in VARIANTS:
        cands = []
        for lr, r in itertools.product(GRID["lr"], GRID["r"]):
            p = f"runs/sweep/{name(v, lr, r, SEL_SEED)}"
            if not os.path.exists(f"{p}/val.npz"):
                print("missing", p)
                continue
            summ = json.load(open(f"ckpt/sweep/{name(v, lr, r, SEL_SEED)}/train_summary.json"))
            ep = [h for h in summ["log_history"] if "eval_loss" in h]
            best_ep = min(ep, key=lambda h: h["eval_loss"])
            for s, st in val_select(p).items():
                cands.append({"variant": v, "lr": lr, "r": r, "score": s, "best_epoch": best_ep["epoch"],
                              "val_loss": best_ep["eval_loss"], **st})
        best = max(cands, key=lambda c: (round(c["J"], 6), c["auroc"]))
        selected[f"lora_{v}"] = best
        zs = [{"variant": v, "score": s, **st} for s, st in val_select(ZEROSHOT[v]).items()]
        selected[f"zs_{v}"] = max(zs, key=lambda c: (round(c["J"], 6), c["auroc"]))
        rows += cands + [{**z, "lr": "-", "r": "-", "best_epoch": "-", "val_loss": float("nan"),
                          "zeroshot": True} for z in zs]

    os.makedirs("results/sweep", exist_ok=True)
    json.dump({"selected": selected, "all": rows}, open("results/sweep/selection.json", "w"), indent=1)
    L = ["| variant | model | lr | r | best epoch | val loss | score | val J | val AUROC | val in-acc | val OOS R |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(rows, key=lambda c: (c["variant"], "zeroshot" in c, -c["J"])):
        sel = any(c is s or (c.get("lr") == s.get("lr") and c.get("r") == s.get("r") and c["score"] == s["score"]
                  and c["variant"] == s["variant"] and ("zeroshot" in c) == (k.startswith("zs")))
                  for k, s in selected.items())
        L.append(f"| {c['variant']} | {'zero-shot' if 'zeroshot' in c else 'LoRA'}{' **(selected)**' if sel else ''} "
                 f"| {c['lr']} | {c['r']} | {c['best_epoch']} | {c['val_loss']:.4f} | {c['score']} "
                 f"| {100 * c['J']:.1f} | {100 * c['auroc']:.1f} | {100 * c['in_acc']:.1f} | {100 * c['oos_recall']:.1f} |")
    open("results/sweep/val_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:2] + [l for l in L if "selected" in l]))
    print(json.dumps(selected, indent=1))


# ----------------------------------------------------------------------------
def test_eval(path, score, tau_from_val=True):
    val, test = ev.Run(path, "val"), ev.Run(path, "test")
    intents = sorted(l for l in test.labels if l != ev.OOS)
    tau = ev.pick_tau(val, intents, CRIT, score)
    m = ev.metrics(test.gold, test.predict("threshold", tau, score), intents)
    m.update({k: v for k, v in ev.ood_metrics(test.gold, test.scores[score]).items()})
    m["aurc"] = ev.coverage_curve(test, score)[2]
    m["tau"] = tau
    if test.has_oos:
        am = ev.metrics(test.gold, test.predict("argmax"), intents)
        m.update({f"argmax_{k}": v for k, v in am.items()})
    return m, test


def bootstrap_ci(path, score, tau, keys, B=1000, seed=0):
    test = ev.Run(path, "test")
    intents = sorted(l for l in test.labels if l != ev.OOS)
    pred = test.predict("threshold", tau, score)
    rng = np.random.default_rng(seed)
    n = len(test.gold)
    samples = {k: [] for k in keys}
    for _ in range(B):
        i = rng.integers(0, n, n)
        m = ev.metrics(test.gold[i], pred[i], intents)
        m.update(ev.ood_metrics(test.gold[i], test.scores[score][i]))
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


KEYS = ["in_acc", "in_macro_f1", "oos_precision", "oos_recall", "oos_f1", "overall_acc", "auroc", "fpr95", "aurc"]


def stage_final(args):
    sel = json.load(open("results/sweep/selection.json"))["selected"]
    jobs = [(v, sel[f"lora_{v}"]["lr"], sel[f"lora_{v}"]["r"], s, ["val", "test"])
            for v in VARIANTS for s in FINAL_SEEDS]
    run_pool(jobs, args.gpus.split(","))

    os.makedirs("results/final", exist_ok=True)
    report, L = {}, []
    L.append("| system | score | tau (val) | " + " | ".join(KEYS) + " |")
    L.append("|---" * (3 + len(KEYS)) + "|")
    for v in VARIANTS:
        # zero-shot: deterministic -> bootstrap 95% CI over test examples
        z = sel[f"zs_{v}"]
        m, _ = test_eval(ZEROSHOT[v], z["score"])
        ci = bootstrap_ci(ZEROSHOT[v], z["score"], m["tau"], [k for k in KEYS if k != "aurc"])
        report[f"zs_{v}"] = {"score": z["score"], "test": m, "ci95": ci}
        L.append(f"| zero-shot {v} | {z['score']} | {m['tau']:.3g} | " + " | ".join(
            f"{100 * m[k]:.1f}" + (f" [{100 * ci[k][0]:.1f}, {100 * ci[k][1]:.1f}]" if k in ci else "")
            for k in KEYS) + " |")
        # LoRA: selected config, mean +- std over seeds
        c = sel[f"lora_{v}"]
        per_seed = {s: test_eval(f"runs/sweep/{name(v, c['lr'], c['r'], s)}", c["score"])[0] for s in FINAL_SEEDS}
        agg = {k: (float(np.mean([per_seed[s][k] for s in FINAL_SEEDS])),
                   float(np.std([per_seed[s][k] for s in FINAL_SEEDS], ddof=1))) for k in KEYS}
        report[f"lora_{v}"] = {"config": {"lr": c["lr"], "r": c["r"]}, "score": c["score"],
                               "per_seed": per_seed, "mean_std": agg}
        taus = ", ".join(f"{per_seed[s]['tau']:.3g}" for s in FINAL_SEEDS)
        L.append(f"| LoRA {v} (lr={c['lr']:g}, r={c['r']}) | {c['score']} | {taus} | " + " | ".join(
            f"{100 * agg[k][0]:.1f} ± {100 * agg[k][1]:.1f}" for k in KEYS) + " |")
    L += ["", "Zero-shot: point estimate [95% bootstrap CI over test examples]. "
          f"LoRA: mean ± std over seeds {FINAL_SEEDS}. Score, tau, lr, r and epoch were all chosen on val "
          f"(criterion: {CRIT}); aurc is lower-is-better, fpr95 lower-is-better."]
    json.dump(report, open("results/final/final_report.json", "w"), indent=1)
    open("results/final/final_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # standard evaluate.py artefacts (predictions, coverage plot, confusions) for seed 42 of each system
    for v in VARIANTS:
        c = sel[f"lora_{v}"]
        for path, score in [(ZEROSHOT[v], sel[f"zs_{v}"]["score"]),
                            (f"runs/sweep/{name(v, c['lr'], c['r'], SEL_SEED)}", c["score"])]:
            subprocess.call([PY, "evaluate.py", path, "--out", f"results/final/{os.path.basename(path)}",
                             "--score", score], cwd=HERE, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "select", "final"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    os.chdir(HERE)
    if args.stage == "grid":
        jobs = [(v, lr, r, SEL_SEED, ["val"]) for v in VARIANTS
                for lr, r in itertools.product(GRID["lr"], GRID["r"])]
        run_pool(jobs, args.gpus.split(","))
    elif args.stage == "select":
        stage_select(args)
    else:
        stage_final(args)


if __name__ == "__main__":
    main()
