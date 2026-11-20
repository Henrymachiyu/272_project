#!/usr/bin/env python3
"""Show model rollouts for a slice of a split: the raw greedy generation plus the
candidate-scoring view (top-3 labels, selected confidence score, decision at the
val-tuned cutoff) taken from the run's saved npz, so it matches the reported results.

    python rollout.py --start 0 --n 10                # first 10 test inputs
    python rollout.py --start 4500 --n 10             # first 10 OOS test inputs
Writes results/rollouts/test_<start>_<start+n>.{json,md}
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clinc  # noqa: E402
import evaluate as ev  # noqa: E402

SYSTEMS_150 = {   # shared team protocol: 150 labels, refuse iff MSP < tau (tau on val by OOS F1)
    "zero-shot 150": {"style": "zeroshot", "run": "runs/zs_150", "adapter": None},
    "LoRA 150 (lr 4e-4, r 8, seed 42)": {
        "style": "lora", "run": "runs/sweep/150_lr0.0004_r8_s42",
        "adapter": "ckpt/sweep/150_lr0.0004_r8_s42/final"},
}
SYSTEMS = {
    "zero-shot 151": {"style": "zeroshot", "run": "runs/zs_151", "adapter": None},
    "LoRA 151 (lr 4e-4, r 32, seed 42, best epoch)": {
        "style": "lora", "run": "runs/sweep/151_lr0.0004_r32_s42",
        "adapter": "ckpt/sweep/151_lr0.0004_r32_s42/final"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--labels", choices=["151", "150"], default="151",
                    help="150 = shared team protocol (MSP cutoff chosen on val by OOS F1)")
    args = ap.parse_args()
    os.chdir(HERE)
    rows = clinc.load_splits()[args.split][args.start:args.start + args.n]
    sel = json.load(open("results/sweep/selection.json"))["selected"]
    final = json.load(open("results/final/final_report.json"))
    out = {"split": args.split, "start": args.start, "systems": {}}

    for name, cfg in (SYSTEMS_150 if args.labels == "150" else SYSTEMS).items():
        if args.labels == "150":
            import common_eval
            score = "msp"
            tau = common_eval.evaluate_split(cfg["run"], "150", "val")[0]["tau"]
        else:
            key = "zs_151" if cfg["style"] == "zeroshot" else "lora_151"
            score = sel[key]["score"]
            tau = final[key]["test"]["tau"] if key == "zs_151" else final[key]["per_seed"]["42"]["tau"]
        run = ev.Run(cfg["run"], args.split)
        assert list(run.texts[args.start:args.start + args.n]) == [t for t, _ in rows]

        tok, model = clinc.load_model(adapter=cfg["adapter"])
        labels = list(run.labels)
        codec = clinc.PromptCodec(tok, cfg["style"], labels)
        recs = []
        for j, (text, gold) in enumerate(rows):
            i = args.start + j
            ids = torch.tensor([codec.prefix_ids + codec.suffix_ids(text)], device="cuda")
            with torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24,
                                   do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id)
            raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
            gen = raw.split("<|im_end|>")[0].strip()
            p = run.P[i]
            top = np.argsort(-p)[:3]
            conf = float(run.scores[score][i])
            pred = run.predict("threshold", tau, score)[i]
            recs.append({
                "idx": i, "text": text, "gold": gold,
                "generation_raw": raw, "generation": clinc.OOS if gen == clinc.OOS_NAME else gen,
                "generation_is_valid_label": gen in {clinc.display(l) for l in labels},
                "scored_argmax": labels[int(p.argmax())],
                "top3": [(labels[t], round(float(p[t]), 4)) for t in top],
                "confidence": {"score": score, "value": round(conf, 6), "tau": round(tau, 6)},
                "final_prediction": str(pred), "correct": bool(pred == gold)})
        out["systems"][name] = recs
        del model
        torch.cuda.empty_cache()

    os.makedirs("results/rollouts", exist_ok=True)
    stem = (f"results/rollouts/{args.split}_{args.start}_{args.start + args.n}"
            + ("_150" if args.labels == "150" else ""))
    json.dump(out, open(stem + ".json", "w"), indent=1)
    L = []
    for name, recs in out["systems"].items():
        c = recs[0]["confidence"]
        L += [f"## {name}", f"confidence score `{c['score']}`, refuse if < tau = {c['tau']}", "",
              "| # | text | gold | raw generation | top-3 (p) | conf | final | ok |", "|---|---|---|---|---|---|---|---|"]
        for r in recs:
            t3 = "<br>".join(f"{l} ({p:.3f})" for l, p in r["top3"])
            raw = r["generation_raw"].replace("|", "\\|").replace("\n", "\\n")
            L.append(f"| {r['idx']} | {r['text']} | {r['gold']} | `{raw}` | {t3} | {r['confidence']['value']:.4f} "
                     f"| {r['final_prediction']} | {'✓' if r['correct'] else '✗'} |")
        L.append("")
    open(stem + ".md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
