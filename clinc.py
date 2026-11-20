"""Shared pieces for the LLM experiments on CLINC150 (OOS-plus).

- data loading (canonical data_oos_plus.json, label list)
- prompt construction (zero-shot prompt with the label list / short LoRA prompt)
- CandidateScorer: exact log P(label | prompt) for every candidate label in a
  single forward pass per batch (all candidates packed after the input with a
  tree-shaped attention mask; the shared prompt prefix is KV-cached once).

Label convention: data label "oos" is shown to the model as "out_of_scope".
"""

import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "data_oos_plus.json")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OOS = "oos"
OOS_NAME = "out_of_scope"

SPLIT_MAP = {"train": ("train", "oos_train"),
             "val": ("val", "oos_val"),
             "test": ("test", "oos_test")}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_splits(path=DATA_PATH):
    """{split: [(text, label), ...]} with label 'oos' for out-of-scope."""
    D = json.load(open(path))
    return {s: [tuple(x) for x in D[a] + D[b]] for s, (a, b) in SPLIT_MAP.items()}


def intent_labels(splits):
    """Sorted list of the 150 in-scope intents."""
    return sorted({l for _, l in splits["train"] if l != OOS})


def label_set(splits, with_oos):
    labs = intent_labels(splits)
    return labs + [OOS] if with_oos else labs


def display(label):
    return OOS_NAME if label == OOS else label


# ----------------------------------------------------------------------------
# prompts.  Every prompt is split into   prefix | suffix(text) | answer
# The prefix is identical for all examples (so it is KV-cached), the answer is
# the label name followed by <|im_end|>.
# ----------------------------------------------------------------------------
SYSTEM = "You are the intent classifier of a voice assistant."
_SENTINEL = "<<<UTTERANCE>>>"


def _zeroshot_user(labels):
    names = ", ".join(display(l) for l in labels)
    if OOS in labels:
        head = ("Classify the user request into exactly one of the intents below. "
                f"If the request is not covered by any of them, answer {OOS_NAME}.")
    else:
        head = "Classify the user request into exactly one of the intents below."
    return (f"{head}\n\nIntents: {names}\n\nRequest: {_SENTINEL}\n"
            "Answer with the intent name only.")


def _lora_user():
    return f"Classify the intent of this request.\nRequest: {_SENTINEL}"


def make_prompt(tokenizer, style, labels):
    """Return (prefix_str, suffix_template) where suffix_template.format(text=...)
    gives the per-example part, ending right where the answer starts."""
    user = _zeroshot_user(labels) if style == "zeroshot" else _lora_user()
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    pre, post = full.split(_SENTINEL)
    return pre, "{text}" + post.replace("{", "{{").replace("}", "}}")


class PromptCodec:
    """Token-level view of a prompt style; used identically for training and scoring."""

    def __init__(self, tokenizer, style, labels):
        self.tok = tokenizer
        self.labels = list(labels)
        pre, self.suffix_tmpl = make_prompt(tokenizer, style, labels)
        self.prefix_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.cand_ids = [self.answer_ids(l) for l in self.labels]

    def suffix_ids(self, text):
        return self.tok(self.suffix_tmpl.format(text=text), add_special_tokens=False)["input_ids"]

    def answer_ids(self, label):
        return self.tok(display(label), add_special_tokens=False)["input_ids"] + [self.im_end]


# ----------------------------------------------------------------------------
# candidate scoring
# ----------------------------------------------------------------------------
class CandidateScorer:
    """Computes sum_t log P(answer_t | prompt, answer_<t) for every candidate.

    Layout of one query row (after the cached prefix of length P):
        [suffix tokens (right-padded to S)] [cand_0 tokens][cand_1 tokens]...
    Candidate tokens attend to the prefix, their own example's suffix and the
    earlier tokens of the same candidate only; their position ids continue
    from the end of the suffix, so this is exactly equivalent to scoring each
    candidate in its own forward pass.
    """

    def __init__(self, model, codec, device):
        self.model, self.codec, self.device = model, codec, device
        self.dtype = next(model.parameters()).dtype
        C = len(codec.cand_ids)

        with torch.no_grad():
            pre = torch.tensor([codec.prefix_ids], device=device)
            out = model(input_ids=pre, use_cache=True)
        pkv = out.past_key_values
        self.prefix_kv = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
        self.P = pre.shape[1]

        # flat candidate block
        ids, off, cid = [], [], []
        for c, toks in enumerate(codec.cand_ids):
            ids += toks
            off += list(range(len(toks)))
            cid += [c] * len(toks)
        L = len(ids)
        self.block_ids = torch.tensor(ids, device=device)
        self.block_off = torch.tensor(off, device=device)
        cid_t = torch.tensor(cid, device=device)
        # block-block mask: same candidate and causal
        self.block_mask = (cid_t[:, None] == cid_t[None, :]) & (self.block_off[:, None] >= self.block_off[None, :])

        # which hidden rows we need logits for: row 0 = last suffix token,
        # then every block token that is not the last token of its candidate.
        pred_rows = []   # block positions used as predictor rows
        row_of = {}
        for q in range(L):
            if q + 1 < L and cid[q + 1] == cid[q]:
                row_of[q] = 1 + len(pred_rows)
                pred_rows.append(q)
        self.pred_block_pos = torch.tensor(pred_rows, device=device, dtype=torch.long)
        tgt_row, tgt_tok, tgt_cand = [], [], []
        for q in range(L):
            tgt_row.append(0 if off[q] == 0 else row_of[q - 1])
            tgt_tok.append(ids[q])
            tgt_cand.append(cid[q])
        self.tgt_row = torch.tensor(tgt_row, device=device)
        self.tgt_tok = torch.tensor(tgt_tok, device=device)
        self.tgt_cand = torch.tensor(tgt_cand, device=device)
        self.C, self.L = C, L

    @torch.no_grad()
    def score(self, suffix_list):
        B, P, L, dev = len(suffix_list), self.P, self.L, self.device
        lens = torch.tensor([len(s) for s in suffix_list], device=dev)
        S = int(lens.max())
        Q = S + L
        pad = self.codec.tok.pad_token_id or 0

        inp = torch.full((B, Q), pad, device=dev, dtype=torch.long)
        for b, s in enumerate(suffix_list):
            inp[b, :len(s)] = torch.tensor(s, device=dev)
        inp[:, S:] = self.block_ids

        ar = torch.arange(S, device=dev)
        pos = torch.empty((B, Q), device=dev, dtype=torch.long)
        pos[:, :S] = P + ar
        pos[:, S:] = P + lens[:, None] + self.block_off[None, :]

        # boolean "may attend" mask, (B, Q, P+Q)
        allow = torch.zeros((B, Q, P + Q), device=dev, dtype=torch.bool)
        allow[:, :, :P] = True
        valid = ar[None, :] < lens[:, None]                      # (B,S) real suffix tokens
        causal = ar[:, None] >= ar[None, :]                      # (S,S)
        allow[:, :S, P:P + S] = causal[None] & valid[:, None, :]
        allow[:, S:, P:P + S] = valid[:, None, :]
        allow[:, S:, P + S:] = self.block_mask[None]
        mask = torch.zeros(allow.shape, device=dev, dtype=self.dtype)
        mask.masked_fill_(~allow, torch.finfo(self.dtype).min)
        mask = mask[:, None]

        from transformers import DynamicCache
        kv = DynamicCache.from_legacy_cache(tuple(
            (k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in self.prefix_kv))

        base = self.model.model if hasattr(self.model, "model") else self.model
        h = base(input_ids=inp, attention_mask=mask, position_ids=pos,
                 past_key_values=kv, use_cache=True).last_hidden_state   # (B,Q,H)

        rows = torch.cat([(lens - 1)[:, None], S + self.pred_block_pos[None, :].expand(B, -1)], 1)
        hs = torch.gather(h, 1, rows[:, :, None].expand(-1, -1, h.shape[-1]))
        logits = self.model.lm_head(hs).float()                  # (B,R,V)
        lse = torch.logsumexp(logits, -1)                         # (B,R)
        tok_lp = logits[:, self.tgt_row, self.tgt_tok] - lse[:, self.tgt_row]   # (B,L)
        out = torch.zeros((B, self.C), device=dev)
        out.index_add_(1, self.tgt_cand, tok_lp)
        return out.cpu()


def load_model(model_id=MODEL_ID, adapter=None, device="cuda", dtype="float32"):
    """Scoring defaults to fp32: in bf16 the label log-probs drift by up to
    ~0.3 nats vs. an exact computation, enough to flip close decisions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa").to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model
