#!/usr/bin/env python3
"""Score every CLINC150 example against every candidate label with Qwen2.5-3B.

Zero-shot (label list in the prompt):
    python score_llm.py --style zeroshot --oos_class --out runs/zs_151
    python score_llm.py --style zeroshot            --out runs/zs_150
LoRA (short prompt, adapter from train_lora.py):
    python score_llm.py --style lora --oos_class --adapter ckpt/lora_151 --out runs/lora_151

--oos_class : candidates are 150 intents + out_of_scope (151-way); otherwise 150.
Multi-GPU   : --gpus 0,1,2,3 launches one worker per GPU on a shard and merges.

Writes <out>/<split>.npz with
    logprobs (N, C) float32  -- sum log P(label tokens | prompt)
    labels   (C,)            -- candidate labels ('oos' = out_of_scope)
    gold     (N,), texts (N,)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinc  # noqa: E402


def run_shard(args, split, shard, nshards):
    import torch
    splits = clinc.load_splits()
    labels = clinc.label_set(splits, args.oos_class)
    rows = splits[split]
    if args.limit:  # debug: an evenly spaced subset (covers in-scope and oos)
        rows = rows[::max(1, len(rows) // args.limit)][:args.limit]
    rows = rows[shard::nshards]
    tok, model = clinc.load_model(adapter=args.adapter, dtype=args.dtype)
    codec = clinc.PromptCodec(tok, args.style, labels)
    scorer = clinc.CandidateScorer(model, codec, "cuda")
    suf = [codec.suffix_ids(t) for t, _ in rows]
    out, t0 = [], time.time()
    for i in range(0, len(rows), args.bs):
        out.append(scorer.score(suf[i:i + args.bs]).numpy())
        if (i // args.bs) % 20 == 0:
            print(f"[{split} {shard}/{nshards}] {i + len(out[-1])}/{len(rows)} "
                  f"{(i + len(out[-1])) / (time.time() - t0):.1f} ex/s", flush=True)
    np.savez(os.path.join(args.out, f"{split}.shard{shard}of{nshards}.npz"),
             logprobs=np.concatenate(out).astype(np.float32),
             labels=np.array(labels), gold=np.array([l for _, l in rows]),
             texts=np.array([t for t, _ in rows]))


def merge(args, split, nshards):
    parts = [np.load(os.path.join(args.out, f"{split}.shard{i}of{nshards}.npz"))
             for i in range(nshards)]
    n = sum(len(p["gold"]) for p in parts)
    C = parts[0]["logprobs"].shape[1]
    lp = np.zeros((n, C), np.float32)
    gold = np.empty(n, dtype=object)
    texts = np.empty(n, dtype=object)
    for i, p in enumerate(parts):          # undo the strided sharding
        lp[i::nshards], gold[i::nshards], texts[i::nshards] = p["logprobs"], p["gold"], p["texts"]
    np.savez(os.path.join(args.out, f"{split}.npz"), logprobs=lp, labels=parts[0]["labels"],
             gold=gold.astype(str), texts=texts.astype(str))
    for i in range(nshards):
        os.remove(os.path.join(args.out, f"{split}.shard{i}of{nshards}.npz"))
    print(f"wrote {args.out}/{split}.npz  ({n} x {C})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["zeroshot", "lora"], required=True)
    ap.add_argument("--oos_class", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N per split")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args._worker:                       # "split:shard:nshards"
        s, k, n = args._worker.split(":")
        run_shard(args, s, int(k), int(n))
        return

    gpus = args.gpus.split(",")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for split in args.splits.split(","):
        procs = []
        for k, g in enumerate(gpus):
            cmd = [sys.executable] + sys.argv + ["--_worker", f"{split}:{k}:{len(gpus)}"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            procs.append(subprocess.Popen(cmd, env=env))
        if any(p.wait() for p in procs):
            sys.exit(f"a worker failed on split {split}")
        merge(args, split, len(gpus))


if __name__ == "__main__":
    main()
