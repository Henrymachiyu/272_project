#!/usr/bin/env python3
"""Local inference server (JSON API) for the web demo: intent classification with OOS refusal.

Hosts Qwen2.5-3B-Instruct under the shared team protocol:
  --style lora --labels 151:     LoRA lr 4e-4, r 16, seed 42; refused iff out_of_scope is the top label
  --style lora --labels 150:     LoRA lr 4e-4, r 8, seed 42;  refused iff top-intent probability < 0.92
  --style zeroshot --labels 151: prompt lists all intents + out_of_scope; refused iff out_of_scope is the top label
  --style zeroshot --labels 150: prompt lists the 150 intents; refused iff top-intent probability < 0.50
(tau chosen on val by OOS F1)

    CUDA_VISIBLE_DEVICES=2 python demo/server.py --style lora --labels 151 --port 8765
    CUDA_VISIBLE_DEVICES=3 python demo/server.py --style lora --labels 150 --port 8766
    CUDA_VISIBLE_DEVICES=2 python demo/server.py --style zeroshot --labels 151 --port 8767
    CUDA_VISIBLE_DEVICES=3 python demo/server.py --style zeroshot --labels 150 --port 8768
API docs: http://localhost:8765/docs   (see demo/API.md)
"""

import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import clinc  # noqa: E402

CONFIGS = {
    ("lora", "151"): {"run": "151_lr0.0004_r16_s42", "tau": None, "report": "lora_151",
                      "model": "Qwen2.5-3B-Instruct + LoRA (r 16, lr 4e-4, seed 42)",
                      "setup": "151 labels: refuse if out_of_scope is the top label"},
    ("lora", "150"): {"run": "150_lr0.0004_r8_s42", "tau": 0.92, "report": "lora_150",
                      "model": "Qwen2.5-3B-Instruct + LoRA (r 8, lr 4e-4, seed 42)",
                      "setup": "150 labels: refuse if the top intent's probability < 0.92 (chosen on val by OOS F1)"},
    ("zeroshot", "151"): {"run": None, "tau": None, "report": "zs_151",
                          "model": "Qwen2.5-3B-Instruct, zero-shot (prompt lists all intents + out_of_scope)",
                          "setup": "151 labels: refuse if out_of_scope is the top label"},
    ("zeroshot", "150"): {"run": None, "tau": 0.50, "report": "zs_150",
                          "model": "Qwen2.5-3B-Instruct, zero-shot (prompt lists the 150 intents)",
                          "setup": "150 labels: refuse if the top intent's probability < 0.50 (chosen on val by OOS F1)"},
}
REFUSAL = "This request looks out of scope — I can't handle it reliably."

app = FastAPI(title="CLINC150 intent classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = {}                      # model state, filled in load()
API_KEY = os.environ.get("CLINC_API_KEY")   # if set, every /api/* route except /api/health needs it
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "missing or wrong X-API-Key header")


AUTH = [Depends(require_key)]
LOCK = threading.Lock()     # one GPU forward at a time


def load(device, style, variant):
    cfg = CONFIGS[(style, variant)]
    splits = clinc.load_splits()
    labels = clinc.label_set(splits, with_oos=variant == "151")
    adapter = os.path.join(ROOT, "ckpt", "sweep", cfg["run"], "final") if cfg["run"] else None
    tok, model = clinc.load_model(adapter=adapter, device=device)
    codec = clinc.PromptCodec(tok, style, labels)
    S.update(tok=tok, model=model, codec=codec, labels=labels, device=device,
             scorer=clinc.CandidateScorer(model, codec, device),
             test=splits["test"], gold={t: l for sp in ("train", "val", "test") for t, l in splits[sp]},
             split_of={t: sp for sp in ("train", "val", "test") for t, _ in splits[sp]})
    rep = json.load(open(os.path.join(ROOT, "results", "common", "common_report.json")))[cfg["report"]]
    S["metrics"] = ({"seed42": rep["per_seed"]["42"], "mean_std": rep["mean_std"]} if style == "lora"
                    else {"test": rep["test"], "ci95": rep["ci95"]})
    S["variant"], S["cfg"] = variant, cfg
    S["bs"] = 32 if style == "lora" else 8     # zero-shot prompt is ~10x longer


def classify(texts, generate=False):
    codec, labels = S["codec"], S["labels"]
    with LOCK:
        t0 = time.time()
        bs = S["bs"]
        lp = torch.cat([S["scorer"].score([codec.suffix_ids(t) for t in texts[i:i + bs]])
                        for i in range(0, len(texts), bs)]).numpy().astype(np.float64)
        gens = [greedy(t) for t in texts] if generate else [None] * len(texts)
        ms = (time.time() - t0) * 1000
    P = np.exp(lp - lp.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    has_oos = clinc.OOS in labels
    oos_i = labels.index(clinc.OOS) if has_oos else None
    tau = S["cfg"]["tau"]
    out = []
    for i, t in enumerate(texts):
        top = np.argsort(-P[i])[:5]
        pred = labels[int(top[0])]
        if tau is not None and P[i, top[0]] < tau:     # 150 labels: MSP cutoff
            pred = clinc.OOS
        gold = S["gold"].get(t.strip())
        out.append({
            "text": t, "prediction": pred, "refused": pred == clinc.OOS,
            "answer": REFUSAL if pred == clinc.OOS else pred,
            "confidence": float(P[i, top[0]]), "tau": tau,
            "p_oos": float(P[i, oos_i]) if has_oos else None,
            "top5": [{"label": clinc.display(labels[j]), "p": float(P[i, j])} for j in top],
            "generation": gens[i],
            "gold": gold, "split": S["split_of"].get(t.strip()),
            "correct": None if gold is None else bool(pred == gold),
        })
    return out, ms


@torch.no_grad()
def greedy(text):
    tok, codec = S["tok"], S["codec"]
    ids = torch.tensor([codec.prefix_ids + codec.suffix_ids(text)], device=S["device"])
    g = S["model"].generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24,
                            do_sample=False, temperature=None, top_p=None, top_k=None,
                            pad_token_id=tok.pad_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False).split("<|im_end|>")[0].strip()


class Query(BaseModel):
    text: str
    generate: bool = False    # also return the model's free greedy generation


class Batch(BaseModel):
    n: int = 50
    oos_share: float = 0.18
    seed: int | None = None


@app.get("/api/health")
def health():
    return {"ok": "scorer" in S}


@app.get("/api/info", dependencies=AUTH)
def info():
    return {"model": S["cfg"]["model"],
            "setup": S["cfg"]["setup"], "tau": S["cfg"]["tau"],
            "n_labels": len(S["labels"]), "metrics": S["metrics"], "refusal": REFUSAL}


@app.post("/api/classify", dependencies=AUTH)
def api_classify(q: Query):
    text = q.text.strip()
    if not text:
        raise HTTPException(400, "empty query")
    if len(text) > 500:
        raise HTTPException(400, "query too long (max 500 characters)")
    res, ms = classify([text], generate=q.generate)
    return {**res[0], "ms": round(ms)}


@app.get("/api/example", dependencies=AUTH)
def api_example(kind: str = "any"):
    rows = S["test"]
    if kind == "oos":
        rows = [r for r in rows if r[1] == clinc.OOS]
    elif kind == "in":
        rows = [r for r in rows if r[1] != clinc.OOS]
    t, l = random.choice(rows)
    return {"text": t, "gold": l}


@app.post("/api/batch", dependencies=AUTH)
def api_batch(b: Batch):
    n = max(10, min(b.n, 500))
    rng = random.Random(b.seed)
    oos = [r for r in S["test"] if r[1] == clinc.OOS]
    ins = [r for r in S["test"] if r[1] != clinc.OOS]
    k = round(n * min(max(b.oos_share, 0.0), 1.0))
    rows = rng.sample(oos, k) + rng.sample(ins, n - k)
    rng.shuffle(rows)
    res, ms = classify([t for t, _ in rows])
    gi = np.array([r["gold"] != clinc.OOS for r in res])
    ok = np.array([r["correct"] for r in res])
    ref = np.array([r["refused"] for r in res])
    summary = {"n": n, "n_in": int(gi.sum()), "n_oos": int((~gi).sum()),
               "in_acc": float(ok[gi].mean()) if gi.any() else None,
               "oos_recall": float(ref[~gi].mean()) if (~gi).any() else None,
               "oos_precision": float((ref & ~gi).sum() / ref.sum()) if ref.any() else None,
               "overall_acc": float(ok.mean()), "ms": round(ms)}
    return {"summary": summary, "items": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--labels", choices=["151", "150"], default="151")
    ap.add_argument("--style", choices=["lora", "zeroshot"], default="lora")
    args = ap.parse_args()
    print(f"loading model ({args.style}, {args.labels} labels) ...", flush=True)
    load(args.device, args.style, args.labels)
    print(f"ready: http://{args.host}:{args.port}  (API key {'required' if API_KEY else 'NOT set'})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
