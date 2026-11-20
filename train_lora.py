#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen2.5-3B-Instruct on CLINC150 (sentence -> intent).

The prompt is the short "lora" style from clinc.py (no label list); loss is on
the answer tokens only (label name + <|im_end|>), so training matches the
candidate scoring in score_llm.py exactly.

    # 151-way: train on all train data incl. the 250 oos examples
    torchrun --nproc_per_node 4 train_lora.py --variant 151 --out ckpt/lora_151
    # 150-way: in-scope data only; OOS handled by a confidence threshold
    torchrun --nproc_per_node 4 train_lora.py --variant 150 --out ckpt/lora_150
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinc  # noqa: E402


class SFTData(torch.utils.data.Dataset):
    def __init__(self, rows, codec):
        self.items = []
        pre = codec.prefix_ids
        for text, label in rows:
            ctx = pre + codec.suffix_ids(text)
            ans = codec.answer_ids(label)
            self.items.append({"input_ids": ctx + ans, "labels": [-100] * len(ctx) + ans})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["151", "150"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16, help="per-device batch size")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=1.0, help="fallback: subsample train")
    ap.add_argument("--keep_checkpoints", action="store_true",
                    help="keep per-epoch checkpoint dirs (default: keep only final/ = best epoch)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    set_seed(args.seed)

    splits = clinc.load_splits()
    with_oos = args.variant == "151"
    labels = clinc.label_set(splits, with_oos)
    keep = (lambda r: True) if with_oos else (lambda r: r[1] != clinc.OOS)
    train_rows = [r for r in splits["train"] if keep(r)]
    val_rows = [r for r in splits["val"] if keep(r)]
    if args.train_frac < 1:
        import random
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[:int(len(train_rows) * args.train_frac)]

    tok = AutoTokenizer.from_pretrained(clinc.MODEL_ID)
    codec = clinc.PromptCodec(tok, "lora", labels)
    train_ds, val_ds = SFTData(train_rows, codec), SFTData(val_rows, codec)

    model = AutoModelForCausalLM.from_pretrained(
        clinc.MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.print_trainable_parameters()
        print(f"train={len(train_ds)} val={len(val_ds)} labels={len(labels)}")
        print("example:", repr(tok.decode(train_ds[0]["input_ids"])))

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine",
        warmup_ratio=0.03, weight_decay=0.0, bf16=True, logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False,
        dataloader_num_workers=2, seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    if trainer.is_world_process_zero():
        final = os.path.join(args.out, "final")
        trainer.model.save_pretrained(final)
        json.dump({**vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
                   "log_history": trainer.state.log_history},
                  open(os.path.join(args.out, "train_summary.json"), "w"), indent=1)
        print("saved", final, "best:", trainer.state.best_model_checkpoint)
        if not args.keep_checkpoints:
            import glob
            import shutil
            for d in glob.glob(os.path.join(args.out, "checkpoint-*")):
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
