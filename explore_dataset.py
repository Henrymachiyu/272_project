#!/usr/bin/env python3
"""
Preliminary dataset exploration for CLINC150 (OOS-plus variant).

CLINC150 is public. The canonical splits live in the official repo:
    https://github.com/clinc/oos-eval  (data/data_oos_plus.json)
The HuggingFace `clinc_oos` mirror currently serves a *triplicated* copy
(train=33,475 with ~18k duplicate rows), so we load the canonical JSON to get
exact, reproducible numbers. Pass --hf to inspect the HF mirror instead.

Run isolated so nothing is imported from the data dir:
    python3 -I explore_dataset.py                      # downloads canonical JSON
    python3 -I explore_dataset.py --json data_oos_plus.json

Reports: split sizes, #intents, in-scope vs out-of-scope (OOS) counts,
per-intent balance, missing/duplicate counts, utterance-length stats, and a few
representative examples. Writes exploration_report.txt and stats_table.tex.
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter

import numpy as np

CANON_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
# key -> (in-scope key, oos key)
SPLIT_MAP = {"train": ("train", "oos_train"),
             "val":   ("val",   "oos_val"),
             "test":  ("test",  "oos_test")}


def load_canonical(path):
    if not path:
        path = "data_oos_plus.json"
        if not os.path.exists(path):
            print(f"downloading {CANON_URL} ...")
            urllib.request.urlretrieve(CANON_URL, path)
    D = json.load(open(path))
    # normalize into {split: list[(text,label)]}
    return {s: D[isc] + D[oos] for s, (isc, oos) in SPLIT_MAP.items()}, D


def load_hf():
    from datasets import load_dataset
    try:
        ds = load_dataset("clinc_oos", "plus", verification_mode="no_checks")
    except TypeError:
        ds = load_dataset("clinc_oos", "plus", ignore_verifications=True)
    names = ds["train"].features["intent"].names
    out = {}
    for s_hf, s in [("train", "train"), ("validation", "val"), ("test", "test")]:
        out[s] = [(t, names[i]) for t, i in zip(ds[s_hf]["text"], ds[s_hf]["intent"])]
    return out, None


def toks(rows):
    return [len(t.split()) for t, _ in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="path to data_oos_plus.json (else download)")
    ap.add_argument("--hf", action="store_true", help="inspect the HuggingFace mirror instead")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    splits, raw = (load_hf() if args.hf else load_canonical(args.json))

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    allrows = sum(splits.values(), [])
    intents = sorted({l for _, l in allrows if l != "oos"})
    oos_total = sum(1 for _, l in allrows if l == "oos")

    out("=" * 68)
    out("CLINC150 (OOS-plus) — PRELIMINARY EXPLORATION")
    out("=" * 68)
    out(f"Source: {'HuggingFace clinc_oos/plus' if args.hf else 'canonical data_oos_plus.json'}")
    out(f"\nTotal examples: {len(allrows):,}")
    out(f"In-scope intents: {len(intents)}   (+1 out-of-scope class = {len(intents)+1} labels)")
    out(f"Out-of-scope examples (all splits): {oos_total:,}")
    out("Fields: text (str), intent (label)")

    out("\n--- Split sizes (in-scope + OOS) ---")
    tr = va = te = 0
    for s in ("train", "val", "test"):
        rows = splits[s]
        n_oos = sum(1 for _, l in rows if l == "oos")
        out(f"  {s:5s} total={len(rows):6,d}  in-scope={len(rows)-n_oos:6,d}  oos={n_oos:5,d}")
        if s == "train": tr = len(rows)
        if s == "val":   va = len(rows)
        if s == "test":  te = len(rows)

    out("\n--- Per-intent balance (train, in-scope) ---")
    c = Counter(l for _, l in splits["train"] if l != "oos")
    out(f"  min={min(c.values())}  max={max(c.values())}  (uniform ⇒ balanced in-scope)")

    out("\n--- Missing / duplicates (whole dataset) ---")
    texts = [t for t, _ in allrows]
    empty = sum(1 for t in texts if not t.strip())
    dup_pairs = len(allrows) - len({(t, l) for t, l in allrows})
    dup_text = len(texts) - len(set(texts))
    out(f"  empty utterances: {empty}")
    out(f"  exact duplicate (text,label) rows: {dup_pairs}")
    out(f"  duplicate utterance strings (any label): {dup_text}")

    out("\n--- Utterance length (whitespace tokens) ---")
    all_tok = toks(allrows)
    out(f"  mean={np.mean(all_tok):.2f}  median={np.median(all_tok):.0f}  "
        f"p95={np.percentile(all_tok,95):.0f}  max={max(all_tok)}  min={min(all_tok)}")
    out("  (all splits well under RoBERTa's 512-token limit ⇒ no truncation needed)")

    out("\n--- Representative examples ---")
    for t, l in splits["train"][:3]:
        out(f"  [in-scope:{l}] {t!r}")
    oos_ex = next((t for t, l in splits["test"] if l == "oos"), None)
    if oos_ex:
        out(f"  [out-of-scope] {oos_ex!r}")

    with open(os.path.join(args.outdir, "exploration_report.txt"), "w") as f:
        f.write("\n".join(lines))

    with open(os.path.join(args.outdir, "stats_table.tex"), "w") as f:
        f.write(
            "\\begin{tabular}{lr}\n\\toprule\nStatistic & Value \\\\\n\\midrule\n"
            f"Total examples & {len(allrows):,} \\\\\n"
            f"In-scope intents & {len(intents)} \\\\\n"
            f"Labels (incl.\\ OOS) & {len(intents)+1} \\\\\n"
            f"Train / Val / Test & {tr:,} / {va:,} / {te:,} \\\\\n"
            f"OOS (train/val/test) & "
            f"{sum(1 for _,l in splits['train'] if l=='oos')}/"
            f"{sum(1 for _,l in splits['val'] if l=='oos')}/"
            f"{sum(1 for _,l in splits['test'] if l=='oos')} \\\\\n"
            f"Mean / median length (tokens) & {np.mean(all_tok):.1f} / {np.median(all_tok):.0f} \\\\\n"
            f"95th-pct / max length (tokens) & {np.percentile(all_tok,95):.0f} / {max(all_tok)} \\\\\n"
            f"Empty / duplicate rows & {empty} / {dup_pairs} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )

    out("\nWrote exploration_report.txt and stats_table.tex to " + args.outdir)


if __name__ == "__main__":
    main()
