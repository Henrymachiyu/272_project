# fine-tune roberta-base on clinc150 (oos-plus) and evaluate out-of-scope (oos) detection
#
# two ways to handle oos requests:
#   --mode 151  oos is a 151st class, trained on the 250 oos_train sentences
#   --mode 150  train on the 150 intents only; refuse (= predict oos) when the top
#               intent probability is below a cutoff picked on val
#
# usage: python roberta_clinc.py --mode 151 --out runs/m151_s0 [--seed 0] [--save_model]
# writes to --out: results.json, labels.json, test_probs.npy, val_probs.npy,
# test_predictions.csv, and model/ with --save_model
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
N_INTENTS = 150   # columns 0-149 of the probabilities are the intents; column 150 is oos in mode 151
DEVICE = "cuda"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["151", "150"], required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "data_oos_plus.json"))
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_model", action="store_true")
    return ap.parse_args()


# ---------- data ----------

def load_data(path, mode):
    # each split is a list of [text, label]; oos sentences are kept in separate oos_* splits
    data = json.load(open(path))

    intents = sorted({label for _, label in data["train"]})
    assert len(intents) == N_INTENTS

    labels = list(intents)
    train = data["train"]
    if mode == "151":
        labels.append("oos")
        train = train + data["oos_train"]

    # val and test always include the oos sentences (in-scope first, then oos)
    val = data["val"] + data["oos_val"]
    test = data["test"] + data["oos_test"]
    return intents, labels, train, val, test


def make_collate(tok, label2id):
    def collate(batch):
        texts = [text for text, _ in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

        # in mode 150, "oos" has no class id. oos sentences then only appear in val/test,
        # where the labels are dropped before the forward pass, so -1 is just a placeholder
        enc["labels"] = torch.tensor([label2id.get(label, -1) for _, label in batch])
        return enc

    return collate


# ---------- prediction and metrics ----------

@torch.no_grad()
def predict_probs(model, loader):
    # softmax probabilities: one row per sentence, one column per label
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits

        out.append(torch.softmax(logits.float(), -1).cpu())

    return torch.cat(out).numpy()


def decide(probs, cutoff, intents, mode):
    # answer = the most likely in-scope intent, or "oos" if refused.
    # refuse when that intent's probability is below the cutoff; in mode 151 also when the oos class wins
    intent_probs = probs[:, :N_INTENTS]
    conf = intent_probs.max(1)
    pred = np.array(intents, dtype=object)[intent_probs.argmax(1)]

    refuse = conf < cutoff
    if mode == "151":
        refuse |= probs.argmax(1) == N_INTENTS

    pred[refuse] = "oos"
    return pred, conf


def oos_score(probs, mode):
    # how oos-like each sentence looks (higher = more oos), used for AUROC
    if mode == "151":
        return probs[:, N_INTENTS]
    return 1 - probs.max(1)


def metrics(gold, probs, cutoff, intents, mode):
    gold = np.array(gold, dtype=object)
    pred, _ = decide(probs, cutoff, intents, mode)
    in_scope = gold != "oos"

    # oos detection as a binary task, oos = positive class
    is_oos = (~in_scope).astype(int)
    said_oos = (pred == "oos").astype(int)
    oos_p, oos_r, oos_f, _ = precision_recall_fscore_support(is_oos, said_oos, average="binary", zero_division=0)

    return {
        # refusing an in-scope sentence counts as wrong
        "in_scope_acc": float((pred[in_scope] == gold[in_scope]).mean()),
        # over the 150 intents on all sentences, so an accepted oos sentence lowers that intent's precision
        "in_scope_macro_f1": float(f1_score(gold, pred, labels=intents, average="macro", zero_division=0)),
        "oos_precision": float(oos_p),
        "oos_recall": float(oos_r),
        "oos_f1": float(oos_f),
        "oos_auroc": float(roc_auc_score(is_oos, oos_score(probs, mode))),
        # plain accuracy over all 151 labels
        "overall_acc_151": float((pred == gold).mean()),
        "cutoff": float(cutoff),
    }


def pick_cutoff(gold, probs, intents, mode):
    # the cutoff (step 0.01) with the best oos f1 on val; ties go to the smaller cutoff
    grid = np.round(np.arange(0.0, 1.0, 0.01), 2)
    f1s = [metrics(gold, probs, c, intents, mode)["oos_f1"] for c in grid]
    return float(grid[int(np.argmax(f1s))])


# ---------- training ----------

def train_one_epoch(model, loader, opt, sched):
    model.train()
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        loss_sum += loss.item()

    return loss_sum / len(loader)


def write_predictions(path, test, probs, labels, cutoff, intents, mode):
    # one row per test sentence, for error analysis
    pred, conf = decide(probs, cutoff, intents, mode)
    top3 = np.argsort(-probs, 1)[:, :3]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "gold", "pred", "conf_top_intent", "p_oos", "top3"])

        for i, (text, gold) in enumerate(test):
            p_oos = f"{probs[i, N_INTENTS]:.4f}" if mode == "151" else ""
            top3_txt = " | ".join(f"{labels[j]}:{probs[i, j]:.3f}" for j in top3[i])
            w.writerow([text, gold, pred[i], f"{conf[i]:.4f}", p_oos, top3_txt])


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    intents, labels, train, val, test = load_data(args.data, args.mode)
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | classes {len(labels)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = make_collate(tok, label2id)
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate, generator=shuffle_gen)
    val_dl = DataLoader(val, batch_size=256, collate_fn=collate)
    test_dl = DataLoader(test, batch_size=256, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
    model = model.to(DEVICE)

    # adamw, 6% linear warmup then linear decay to 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    # train, keep the epoch with the best val score.
    # no cutoff during training: mode 151 is scored by plain 151-way accuracy,
    # mode 150 by in-scope accuracy (it cannot predict oos without a cutoff)
    val_gold = [label for _, label in val]
    best_score, best_epoch, best_state = -1, None, None
    log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, opt, sched)
        val_m = metrics(val_gold, predict_probs(model, val_dl), 0.0, intents, args.mode)
        score = val_m["overall_acc_151"] if args.mode == "151" else val_m["in_scope_acc"]
        sec = time.time() - t0

        log.append({"epoch": epoch, "train_loss": train_loss, "val": val_m, "sec": sec})
        print(f"ep {epoch} loss {train_loss:.4f} val in-acc {val_m['in_scope_acc']:.4f} "
              f"overall {val_m['overall_acc_151']:.4f} oos-rec {val_m['oos_recall']:.4f} ({sec:.0f}s)")

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if args.save_model:
        model.save_pretrained(os.path.join(args.out, "model"))
        tok.save_pretrained(os.path.join(args.out, "model"))

    # final evaluation: pick the cutoff on val, then apply it to test
    val_probs = predict_probs(model, val_dl)
    test_probs = predict_probs(model, test_dl)
    test_gold = [label for _, label in test]
    cutoff = pick_cutoff(val_gold, val_probs, intents, args.mode)

    results = {
        "mode": args.mode,
        "best_epoch": best_epoch,
        "args": vars(args),
        "test_no_cutoff": metrics(test_gold, test_probs, 0.0, intents, args.mode),
        "test_val_cutoff": metrics(test_gold, test_probs, cutoff, intents, args.mode),
        "val_val_cutoff": metrics(val_gold, val_probs, cutoff, intents, args.mode),
        "log": log,
    }
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"))
    np.save(os.path.join(args.out, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.out, "val_probs.npy"), val_probs)
    write_predictions(os.path.join(args.out, "test_predictions.csv"), test, test_probs, labels, cutoff,
                      intents, args.mode)

    print("best epoch", best_epoch, "| val cutoff", cutoff)
    for key in ["test_no_cutoff", "test_val_cutoff"]:
        print(key, {name: round(v, 4) for name, v in results[key].items()})


if __name__ == "__main__":
    main()
