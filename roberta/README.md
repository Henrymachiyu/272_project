# RoBERTa-base on CLINC150 (OOS-plus)

Note: the code and this README were written with Claude Code (AI assistant) at Junwei's request. Claude Code later reformatted the code to follow Junwei's coding style, and then restructured it for readability (clearer names, small functions, comments). Both passes were checked against the previous version: identical seed table, error analysis and figure, and bit-identical predicted probabilities in one-epoch training runs of both modes.

## Setup
- data: `data/data_oos_plus.json` (original JSON from github.com/clinc/oos-eval, not included here; counts match the proposal: train 15,000 + 250 oos, val 3,000 + 100, test 4,500 + 1,000)
- model: `FacebookAI/roberta-base`, full fine-tuning, AdamW lr 3e-5, wd 0.01, batch 32, 6% warmup + linear decay, bf16, max 64 tokens, 5 epochs, seed 0
- epoch picked on val (151: overall 151-way acc; 150: in-scope acc)
- two OOS methods:
  - **151-class**: `oos` is a class, trained on the 250 oos_train sentences
  - **150-class + cutoff**: trained on the 150 intents only; predict `oos` if the top probability < cutoff
- cutoff: the value with the best OOS F1 on val (grid step 0.01). For the 151 model, we also apply it on top of the argmax.
- OOS score for AUROC: p(oos) for 151, 1 - max prob for 150

## Run
```
pip install -r requirements.txt
mkdir -p data && curl -L -o data/data_oos_plus.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json
python roberta_clinc.py --mode 151 --out runs/m151_s0 --save_model
python roberta_clinc.py --mode 150 --out runs/m150_s0 --save_model
python analyze.py s0          # figs/roberta_cutoff_curves_s0.png + error_analysis_s0.md
```
One run takes about 2 min on one GPU (about 11 s per epoch plus loading). `analyze.py` needs the `test_probs.npy` files that training writes, so run it after training.

## Test results over 3 seeds (5 epochs, seeds 0/1/2, mean ± std)

| method | in-scope acc | in-scope macro-F1 | OOS precision | OOS recall | OOS F1 | OOS AUROC | 151-way acc |
|---|---|---|---|---|---|---|---|
| 151-class (argmax) | 96.53 ± 0.24 | 92.89 ± 0.33 | 98.09 ± 0.71 | 61.70 ± 1.01 | 75.75 ± 0.84 | **97.18 ± 0.56** | 90.19 ± 0.38 |
| 151-class + val cutoff | 96.53 ± 0.24 | 92.95 ± 0.39 | 98.07 ± 0.75 | 62.30 ± 1.57 | **76.18 ± 1.08** | 97.18 ± 0.56 | **90.30 ± 0.48** |
| 150-class, no cutoff | **96.67 ± 0.10** | 87.82 ± 0.13 | – | 0 | 0 | 96.14 ± 0.05 | 79.09 ± 0.08 |
| 150-class + val cutoff | 95.84 ± 0.57 | **93.12 ± 0.87** | 87.99 ± 4.10 | **62.70 ± 10.70** | 72.65 ± 6.32 | 96.14 ± 0.05 | 89.81 ± 1.50 |

- cutoffs picked on val: 151 = 0.17 / 0.00 / 0.00, 150 = 0.79 / 0.79 / 0.52
- same seed, 151 + cutoff minus 150 + cutoff: AUROC +0.40 / +1.44 / +1.27, in-scope acc +0.89 / +0.96 / +0.22, OOS precision +10.3 / +13.7 / +6.2, OOS recall -2.9 / -10.3 / +12.0, OOS F1 +1.6 / -2.2 / +11.2
- full table: `results_seeds.md` (`python summarize_seeds.py`), seeds 1-2 runs via `run_seeds.sh`

## Test results, seed 0 only (the run used for the figure and error analysis)

| method | in-scope acc | in-scope macro-F1 | OOS precision | OOS recall | OOS F1 | OOS AUROC | 151-way acc |
|---|---|---|---|---|---|---|---|
| 151-class (argmax) | 96.69 | 93.12 | 97.33 | 61.90 | 75.67 | 96.54 | 90.36 |
| 151-class + cutoff 0.17 | 96.69 | 93.29 | 97.25 | 63.70 | **76.98** | 96.54 | **90.69** |
| 150-class, no cutoff | **96.78** | 87.97 | – | 0.00 | 0.00 | 96.14 | 79.18 |
| 150-class + cutoff 0.79 | 95.80 | **93.55** | 86.95 | **66.60** | 75.42 | 96.14 | 90.49 |

- in-scope acc is computed on the 4,500 in-scope sentences; predicting `oos` for one counts as wrong
- in-scope macro-F1 is over the 150 intents on all 5,500 test sentences, so OOS sentences that are accepted as an intent lower that intent's precision (that is why "150, no cutoff" is low)
- 10-epoch check (`runs/m15*_e10`): no consistent gain (val +0.2 pt for 151, -0.1 pt for 150; test OOS F1 75.2 and 70.6), so we keep 5 epochs
## What we see
- Supported intents are easy for RoBERTa (about 96.5-96.7% over 3 seeds). The hard part is OOS: about 1/3 of OOS sentences are still accepted as some intent (seed 0: 363 of 1,000 for the 151 model, 334 for 150 + cutoff).
- Training on the 250 OOS examples (151-class) is better than a cutoff alone in every seed on AUROC (+0.4 to +1.4), in-scope accuracy (+0.2 to +1.0) and OOS precision (+6 to +14). It is also much more stable: OOS F1 76.2 ± 1.1 vs 72.7 ± 6.3.
- The 150 + cutoff method is unstable because of the cutoff, not the model: its AUROC barely moves across seeds (96.14 ± 0.05), but the cutoff picked on the 100 val OOS sentences jumps between 0.52 and 0.79, so OOS recall goes from 50.6 to 70.9. With few OOS examples for tuning, a confidence cutoff is hard to set.
- For the 151 model, the val-picked cutoff is 0 in 2 of 3 seeds, so the cutoff adds little on top of the oos class. The right panel of the figure (seed 0) shows that the 151 model could still trade some in-scope accuracy for recall (for example, about 80% recall around cutoff 0.7). We did not pick the cutoff on test.
- Common confusions are near-duplicate intents in the same topic: apr -> interest_rate (6), redeem_rewards -> rewards_balance (5), calendar -> calendar_update (4), pay_bill -> bill_due (3).
- Some errors look like label noise in CLINC150, not model mistakes:
  - OOS-labeled but in scope: "look up the conversion rate for the euro to dollar exchange" (model: exchange_rate, 0.99), "call an uber to take me to the closest grocery store" (model: uber, 0.99)
  - pin_change-labeled: "i would like to change the name on my credit card account" (model: change_ai_name)
- OOS sentences accepted with high confidence are often "near-topic" requests: Amazon product reviews -> restaurant_reviews (0.99), mortgage rates -> interest_rate, bus stop -> directions.

## Files
- `roberta_clinc.py`: training and evaluation (`--mode 151|150`); `analyze.py`: figure + error analysis; `summarize_seeds.py`: the 3-seed table; `run_seeds.sh`: seeds 1-2
- `runs/<run>/results.json` (all metrics + per-epoch val log) and `labels.json` for every run: `m151_s0`, `m150_s0` (seed 0), `m15*_s1`, `m15*_s2` (seeds 1-2), `m15*_e10` (10-epoch check)
- `runs/m151_s0`, `runs/m150_s0`: also `test_predictions.csv` (text, gold, pred, confidence, top-3)
- not in this repo (too large, written by training): `model/` (HF format, about 480 MB per run, with `--save_model`) and `test_probs.npy` / `val_probs.npy` (rows = test/val order: in-scope then oos; columns = `labels.json`)
- `test_0_10_roberta.md`: both models on the first 10 test sentences, in the team's comparison format
- `figs/roberta_cutoff_curves_s0.png`: left = accuracy on answered inputs vs coverage, right = in-scope acc / OOS recall vs cutoff
- `error_analysis_s0.md`: top confused pairs, OOS accepted as which intents, in-scope intents refused most

## Loading a trained model
The team demo uses the 150-class model with cutoff 0.79 (seed 0).
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "runs/m150_s0/model"   # written by: python roberta_clinc.py --mode 150 --out runs/m150_s0 --save_model
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()

enc = tok("how much has the dow changed today", return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, -1)[0]

top = int(probs.argmax())     # most likely intent
label = "oos" if probs[top] < 0.79 else model.config.id2label[top]

# 151-class model (runs/m151_s0/model, cutoff 0.17): take top = int(probs[:150].argmax()),
# and refuse if probs.argmax() == 150 (the oos class wins) or probs[top] < 0.17
```
