# roberta-base on clinc150 (oos-plus)
# mode 151: oos is the 151st class (train uses oos_train)
# mode 150: 150 intents only, oos = top prob below a cutoff picked on val
import argparse
import json
import os
import random
import time
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

here = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["151", "150"], required=True)
ap.add_argument("--data", default=os.path.join(here, "data", "data_oos_plus.json"))
ap.add_argument("--model", default="FacebookAI/roberta-base")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save_model", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda"


# load data
d = json.load(open(args.data))

intents = sorted({y for _, y in d["train"]})
assert len(intents) == 150

labels = list(intents)
if args.mode == "151":
    labels.append("oos")

lab2id = {}
for i, l in enumerate(labels):
    lab2id[l] = i

train = d["train"]
if args.mode == "151":
    train = train + d["oos_train"]

val = d["val"] + d["oos_val"]
test = d["test"] + d["oos_test"]
print(f"train {len(train)} | val {len(val)} | test {len(test)} | classes {len(labels)}")

tok = AutoTokenizer.from_pretrained(args.model)


def collate(batch):
    texts = [t for t, _ in batch]
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

    # oos gold gets -1 in mode 150 (never trained on)
    ids = [lab2id.get(y, -1) for _, y in batch]
    enc["labels"] = torch.tensor(ids)
    return enc


g = torch.Generator().manual_seed(args.seed)
train_dl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate, generator=g)
val_dl = DataLoader(val, batch_size=256, collate_fn=collate)
test_dl = DataLoader(test, batch_size=256, collate_fn=collate)

id2label = dict(enumerate(labels))
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels), id2label=id2label, label2id=lab2id)
model = model.to(dev)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * len(train_dl)
warmup = int(0.06 * steps)
sched = get_linear_schedule_with_warmup(opt, warmup, steps)


@torch.no_grad()
def get_probs(dl):
    model.eval()
    out = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**b).logits

        probs = torch.softmax(logits.float(), -1)
        out.append(probs.cpu())

    return torch.cat(out).numpy()


def decide(probs, cutoff):
    # top in-scope intent, refuse if its prob < cutoff (mode 151 also refuses if oos wins)
    p_in = probs[:, :150]
    top = p_in.argmax(1)
    conf = p_in.max(1)
    pred = np.array([intents[i] for i in top], dtype=object)

    refuse = conf < cutoff
    if args.mode == "151":
        refuse |= probs.argmax(1) == 150

    pred[refuse] = "oos"
    return pred, conf


def oos_score(probs):
    # higher = more likely oos
    if args.mode == "151":
        return probs[:, 150]
    else:
        return 1 - probs.max(1)


def metrics(gold, probs, cutoff):
    gold = np.array(gold, dtype=object)
    pred, conf = decide(probs, cutoff)
    ins = gold != "oos"

    # oos as the positive class
    is_oos = (~ins).astype(int)
    said_oos = (pred == "oos").astype(int)
    p, r, f, _ = precision_recall_fscore_support(is_oos, said_oos, average="binary", zero_division=0)

    in_acc = (pred[ins] == gold[ins]).mean()
    macro_f1 = f1_score(gold, pred, labels=intents, average="macro", zero_division=0)
    auroc = roc_auc_score(is_oos, oos_score(probs))
    overall = (pred == gold).mean()

    return {
        "in_scope_acc": float(in_acc),
        "in_scope_macro_f1": float(macro_f1),
        "oos_precision": float(p),
        "oos_recall": float(r),
        "oos_f1": float(f),
        "oos_auroc": float(auroc),
        "overall_acc_151": float(overall),
        "cutoff": float(cutoff),
    }


def pick_cutoff(gold, probs):
    # cutoff with the best oos f1 on val
    grid = np.round(np.arange(0.0, 1.0, 0.01), 2)
    scores = []
    for c in grid:
        scores.append(metrics(gold, probs, c)["oos_f1"])

    best_i = int(np.argmax(scores))
    return float(grid[best_i])


# train, keep the epoch with the best val score
val_gold = [y for _, y in val]
best = -1
log = []
best_path = os.path.join(args.out, "best.pt")

for ep in range(args.epochs):
    model.train()
    t0 = time.time()
    tot = 0.0

    for b in train_dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**b).loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        tot += loss.item()

    vp = get_probs(val_dl)

    # mode 151 selects on plain argmax; mode 150 on in-scope acc (cutoff is set later)
    vm = metrics(val_gold, vp, 0.0)
    if args.mode == "151":
        score = vm["overall_acc_151"]
    else:
        score = vm["in_scope_acc"]

    train_loss = tot / len(train_dl)
    sec = time.time() - t0
    log.append({"epoch": ep + 1, "train_loss": train_loss, "val": vm, "sec": sec})
    print(f"ep {ep+1} loss {train_loss:.4f} val in-acc {vm['in_scope_acc']:.4f} "
          f"overall {vm['overall_acc_151']:.4f} oos-rec {vm['oos_recall']:.4f} ({time.time()-t0:.0f}s)")

    if score > best:
        best = score
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, best_path)
        best_ep = ep + 1


# reload best
model.load_state_dict(torch.load(best_path))
os.remove(best_path)

if args.save_model:
    model_dir = os.path.join(args.out, "model")
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

vp = get_probs(val_dl)
tp = get_probs(test_dl)
test_gold = [y for _, y in test]
cut = pick_cutoff(val_gold, vp)

res = {
    "mode": args.mode,
    "best_epoch": best_ep,
    "args": vars(args),
    "test_no_cutoff": metrics(test_gold, tp, 0.0),
    "test_val_cutoff": metrics(test_gold, tp, cut),
    "val_val_cutoff": metrics(val_gold, vp, cut),
    "log": log,
}
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
np.save(os.path.join(args.out, "test_probs.npy"), tp)
np.save(os.path.join(args.out, "val_probs.npy"), vp)
json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"))


# per-example predictions for error analysis
pred, conf = decide(tp, cut)
top3 = np.argsort(-tp, 1)[:, :3]

with open(os.path.join(args.out, "test_predictions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "gold", "pred", "conf_top_intent", "p_oos", "top3"])

    for i, (t, y) in enumerate(test):
        p_oos = ""
        if args.mode == "151":
            p_oos = f"{tp[i, 150]:.4f}"

        top3_txt = []
        for j in top3[i]:
            top3_txt.append(f"{labels[j]}:{tp[i, j]:.3f}")

        w.writerow([t, y, pred[i], f"{conf[i]:.4f}", p_oos, " | ".join(top3_txt)])

print("best epoch", best_ep, "| val cutoff", cut)
for k in ["test_no_cutoff", "test_val_cutoff"]:
    rounded = {}
    for a, b in res[k].items():
        rounded[a] = round(b, 4)
    print(k, rounded)
