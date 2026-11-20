#!/usr/bin/env python3
"""Sanity check: does candidate scoring describe what the model actually says?

For a stratified test subset, decode greedily with the *same* prompt and
compare the free-form answer with the candidate-scoring argmax stored in
runs/<name>/test.npz.  Reports: share of generations that are a valid label,
agreement with the scored argmax, and the generated answer's own sequence
log-prob vs. the scored top label.

    python check_generation.py --run runs/zs_151 --style zeroshot --oos_class
    python check_generation.py --run runs/lora_151 --style lora --oos_class --adapter ckpt/lora_151/final
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--style", choices=["zeroshot", "lora"], required=True)
    ap.add_argument("--oos_class", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=600, help="examples (half in-scope, half oos)")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    d = np.load(os.path.join(args.run, "test.npz"))
    labels = [str(x) for x in d["labels"]]
    gold, texts = d["gold"].astype(str), d["texts"].astype(str)
    scored = np.array(labels)[d["logprobs"].argmax(1)]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(gold != "oos")[0], args.n // 2, replace=False),
                          rng.choice(np.where(gold == "oos")[0], args.n // 2, replace=False)])

    tok, model = clinc.load_model(adapter=args.adapter)
    codec = clinc.PromptCodec(tok, args.style, labels)
    name2label = {clinc.display(l): l for l in labels}
    gen = []
    for i in range(0, len(idx), args.bs):
        seqs = [codec.prefix_ids + codec.suffix_ids(texts[j]) for j in idx[i:i + args.bs]]
        n = max(map(len, seqs))
        inp = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in seqs], device="cuda")
        att = (torch.arange(n, device="cuda")[None] >= torch.tensor([n - len(s) for s in seqs], device="cuda")[:, None]).long()
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=16, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        gen += [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]

    gen_label = [name2label.get(g) for g in gen]
    valid = np.array([g is not None for g in gen_label])
    agree = np.array([g == s for g, s in zip(gen_label, scored[idx])])
    res = {"run": args.run, "n": int(len(idx)),
           "valid_label_rate": float(valid.mean()),
           "agreement_with_scored_argmax": float(agree.mean()),
           "agreement_in_scope_gold": float(agree[: args.n // 2].mean()),
           "agreement_oos_gold": float(agree[args.n // 2:].mean()),
           "gen_accuracy": float(np.mean([g == y for g, y in zip(gen_label, gold[idx])])),
           "scored_accuracy": float(np.mean(scored[idx] == gold[idx])),
           "invalid_examples": [(texts[j], g) for j, g, v in zip(idx, gen, valid) if not v][:15],
           "disagreements": [(texts[j], g, scored[j]) for j, g, a, v in zip(idx, gen, agree, valid)
                             if v and not a][:15]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(args.run, "generation_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
